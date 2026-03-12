import os, sys, json
import pickle
from autogen.coding import CodeBlock
from autogen.coding.jupyter import JupyterCodeExecutor, LocalJupyterServer
import ast, re

# add the tools directory to the path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# for each dialogue, we will have a new code executor
class CodeExecutor:
    def __init__(
        self, 
        working_dir: str = "",
        use_vision_tools: bool = False,
        ):
        self.working_dir = working_dir
        # Self-evolving meta-tool probe / escalation state (per executor/task)
        self._meta_tool_use_preflight_done = False
        self._meta_tool_use_preflight_result = None
        self._pending_meta_escalation = False
        self._meta_escalation_hint_done = False
        self._consecutive_visual_tool_calls = 0
        self._consecutive_low_quality_visual = 0
        self._last_visual_tool_low_quality = False
        self._meta_escalation_reason = None
        self._meta_event_log = os.path.join(self.working_dir, 'meta_tools_escalation_events.jsonl')
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
            
        # set up the server
        self.server = LocalJupyterServer()
            
        # set up the jupyter executor
        self.executor = JupyterCodeExecutor(self.server, output_dir=self.working_dir)
        
        # initialize the environment
        self.init_env(use_vision_tools)
        
    def result_processor(self, result):
        # Change an IPythonCodeResult object to a string, and the list of files
        # If the execution failed, the string is the error message.
        # If the execution is successful, the string is the output of the code execution.
        # In the string, all embeded PIL images are replaced by their file paths, using html img tag.
        # The list of files are the paths of the images.
        
        # process error message
        def parse_error_message(error):
            # Find the index where the list starts, indicated by `['`
            list_start_index = error.find("['")
            
            # The first part before the list is the initial error message
            initial_error = error[:list_start_index].strip()
            
            # The second part is the list of strings, which starts from `['` and goes to the end of the string
            traceback_list_str = error[list_start_index:]
            
            # Use ast.literal_eval to safely evaluate the string representation of the list
            # This method is safer than eval and can handle Python literals properly
            try:
                traceback_list = ast.literal_eval(traceback_list_str)
            except SyntaxError as e:
                print("Error parsing the list: ", e)
                traceback_list = []
                
            # Remove ANSI escape sequences
            ansi_escape = re.compile(r'\x1b\[.*?m')
            traceback_list = [ansi_escape.sub('', line) for line in traceback_list]
            
            return initial_error + "\n\n" + "\n".join(traceback_list)
        
        
        exit_code = result.exit_code
        
        file_paths = result.output_files
        output_str = result.output
        output_lines = output_str.split("\n")
        
        if len(file_paths) > 0:
            output_lines = output_lines[:-2*len(file_paths)]
            
        # if execution succeeded, replace PIL images with their file paths
        if exit_code == 0:
            new_str = ""
            image_idx = 0
            
            for line in output_lines:
                if line.startswith("<PIL."):
                    if image_idx < len(file_paths):
                        new_str += f"<img src='{file_paths[image_idx]}'>"
                        image_idx += 1
                else:
                    new_str += line
                new_str += "\n"
            
            # add the remaining images
            for file_idx, file in enumerate(file_paths):
                if file_idx >= image_idx:
                    new_str += f"<img src='{file}'>"
                    new_str += "\n"
                
            return exit_code, new_str, file_paths
        
        # if execution failed, parse the error message
        else:
            error_msg = parse_error_message(output_str)
            return exit_code, error_msg, file_paths
    

    def _append_meta_event(self, event: dict):
        """Best-effort JSONL event logging for debugging self-evolving tool triggers."""
        try:
            os.makedirs(self.working_dir, exist_ok=True)
            with open(self._meta_event_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _code_uses_visual_tools(self, code: str) -> bool:
        if not isinstance(code, str):
            return False
        visual_pat = re.compile(
            r"\b(detection|sliding_window_detection|segment_and_mark|depth|crop_image|zoom_in_image_by_bbox|overlay_images)\s*\("
        )
        return bool(visual_pat.search(code))

    def _code_uses_meta_tools(self, code: str) -> bool:
        if not isinstance(code, str):
            return False
        meta_pat = re.compile(
            r"\b(inspect_tools|propose_tool|test_candidate_tool|commit_candidate_tool|disable_tool)\s*\("
        )
        return bool(meta_pat.search(code))

    def _maybe_meta_tool_preflight_on_tool_use(self, code: str):
        """
        Auto-trigger a one-time inspect_tools() preflight when the model starts using
        standard visual tools. Behavior: prepend inspect_tools() to the SAME code
        execution so the output is visible in the current observation.
        Returns possibly modified code.
        """
        try:
            if os.environ.get("VSK_TOOL_EVOLVE", "0") != "1":
                return code
            if getattr(self, "_meta_tool_use_preflight_done", False):
                return code
            if not isinstance(code, str):
                return code
            if (not self._code_uses_visual_tools(code)) or self._code_uses_meta_tools(code):
                return code

            self._meta_tool_use_preflight_done = True  # avoid repeat

            preflight_json = os.path.join(self.working_dir, "meta_tools_on_tool_use.json")
            preflight_err = os.path.join(self.working_dir, "meta_tools_on_tool_use_error.txt")

            preflight_code = (
                "import json\n"
                "try:\n"
                "    _vsk_tool_inventory = inspect_tools(include_dynamic=True)\n"
                "    print('[VSK_META_TOOL_USE_PREFLIGHT] inspect_tools ok')\n"
                "    print(json.dumps(_vsk_tool_inventory, ensure_ascii=False))\n"
                f"    with open({preflight_json!r}, 'w', encoding='utf-8') as _f:\n"
                "        json.dump(_vsk_tool_inventory, _f, ensure_ascii=False, indent=2)\n"
                "except Exception as _e:\n"
                "    print('[VSK_META_TOOL_USE_PREFLIGHT_ERROR]', repr(_e))\n"
                f"    with open({preflight_err!r}, 'w', encoding='utf-8') as _f:\n"
                "        _f.write(repr(_e))\n"
            )

            self._meta_tool_use_preflight_result = "prefixed"
            self._append_meta_event({
                "event": "meta_preflight_prefixed",
                "working_dir": self.working_dir,
            })
            return preflight_code + "\n" + code

        except Exception as e:
            self._meta_tool_use_preflight_result = f"error: {e}"
            try:
                with open(os.path.join(self.working_dir, "meta_tools_on_tool_use_error.txt"), "w", encoding="utf-8") as _f:
                    _f.write(repr(e))
            except Exception:
                pass
            self._append_meta_event({
                "event": "meta_preflight_error",
                "error": repr(e),
                "working_dir": self.working_dir,
            })
            return code

    def _maybe_meta_escalation_hint_prefix(self, code: str):
        """
        If prior visual-tool observations looked low-quality (empty boxes / no progress),
        prepend a strong hint that nudges the model into propose_tool -> test -> commit.
        This keeps the hint in the current observation stream without changing user prompts.
        """
        try:
            if os.environ.get("VSK_TOOL_EVOLVE", "0") != "1":
                return code
            if not isinstance(code, str):
                return code
            if self._meta_escalation_hint_done:
                return code
            if not self._pending_meta_escalation:
                return code
            if self._code_uses_meta_tools(code):
                return code
            if not self._code_uses_visual_tools(code):
                return code

            hint_json = os.path.join(self.working_dir, "meta_tools_escalation_hint_triggered.json")
            reason = self._meta_escalation_reason or "low_quality_or_repeated_visual_tool_output"

            hint_code = (
                "import json\n"
                "print('[VSK_META_ESCALATION_HINT] Visual tool results were low-quality or repeated without progress.')\n"
                "print('[VSK_META_ESCALATION_HINT] Switch to self-evolving meta-tools now: inspect_tools -> propose_tool -> test_candidate_tool -> commit_candidate_tool (if ok).')\n"
                "print('[VSK_META_ESCALATION_HINT] Prefer a SMALL wrapper helper that reuses existing tools (retry / sliding-window / zoom / normalize output).')\n"
                f"_vsk_meta_escalation = {{'event':'meta_escalation_hint','reason':{reason!r}}}\n"
                f"with open({hint_json!r}, 'w', encoding='utf-8') as _f:\n"
                "    json.dump(_vsk_meta_escalation, _f, ensure_ascii=False, indent=2)\n"
            )

            self._meta_escalation_hint_done = True
            self._append_meta_event({
                "event": "meta_escalation_hint_prefixed",
                "reason": reason,
                "working_dir": self.working_dir,
            })
            return hint_code + "\n" + code
        except Exception as e:
            self._append_meta_event({
                "event": "meta_escalation_hint_error",
                "error": repr(e),
                "working_dir": self.working_dir,
            })
            return code

    def _detect_low_quality_visual_output(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        t = text.lower()
        patterns = [
            r"traceback",
            r"execution failed",
            r"\bexception\b",
            r"\berror\b",
            r"\bno boxes?\b",
            r"\b0 boxes?\b",
            r"detected boxes\s*[:=]?\s*\[\s*\]",
            r"\bboxes?\s*[:=]\s*\[\s*\]",
            r"\bbboxes\s*[:=]\s*\[\s*\]",
            r"possible_boxes\s*[:=]\s*\[\s*\]",
            r"\bnot found\b",
            r"\bempty result\b",
        ]
        return any(re.search(p, t) for p in patterns)

    def _update_meta_evolve_state_after_execute(self, executed_code: str, ret):
        """
        Update per-task counters and decide whether to inject a meta-escalation hint
        before the next repeated visual-tool attempt.
        """
        try:
            is_visual = self._code_uses_visual_tools(executed_code)
            is_meta = self._code_uses_meta_tools(executed_code)

            if is_meta:
                self._pending_meta_escalation = False
                self._meta_escalation_reason = None
                self._append_meta_event({"event": "meta_tool_call_observed", "working_dir": self.working_dir})
                return

            if not is_visual:
                self._consecutive_visual_tool_calls = 0
                return

            self._consecutive_visual_tool_calls += 1

            obs_text = ""
            if isinstance(ret, tuple) and len(ret) >= 2:
                obs_text = ret[1] or ""

            low_quality = self._detect_low_quality_visual_output(obs_text)
            self._last_visual_tool_low_quality = low_quality

            if low_quality:
                self._consecutive_low_quality_visual += 1
            else:
                self._consecutive_low_quality_visual = 0

            trigger_reason = None
            if low_quality:
                trigger_reason = "low_quality_visual_output"
            elif self._consecutive_visual_tool_calls >= 2:
                trigger_reason = "repeated_visual_tool_calls"

            if trigger_reason and (not self._meta_escalation_hint_done):
                self._pending_meta_escalation = True
                self._meta_escalation_reason = trigger_reason
                self._append_meta_event({
                    "event": "meta_escalation_pending",
                    "reason": trigger_reason,
                    "consecutive_visual_tool_calls": self._consecutive_visual_tool_calls,
                    "consecutive_low_quality_visual": self._consecutive_low_quality_visual,
                    "working_dir": self.working_dir,
                })
        except Exception as e:
            self._append_meta_event({
                "event": "meta_state_update_error",
                "error": repr(e),
                "working_dir": self.working_dir,
            })

    def execute(self, code: str):
        # 1) On first standard visual-tool use, prepend inspect_tools() so inventory is visible.
        code = self._maybe_meta_tool_preflight_on_tool_use(code)
        # 2) After repeated / low-quality visual outputs, prepend a one-time escalation hint.
        code = self._maybe_meta_escalation_hint_prefix(code)

        self.executor._jupyter_kernel_client = self.executor._jupyter_client.get_kernel_client(self.executor._kernel_id)
        execution_result = self.executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python",
                        code=code),
            ]
        )
        ret = self.result_processor(execution_result)
        self._update_meta_evolve_state_after_execute(code, ret)
        return ret
    
    def init_env(self, use_vision_tools):
        init_code = ("import sys\n"
                     "from PIL import Image\n"
                     "from IPython.display import display\n"
                     f"parent_dir = '{parent_dir}'\n"
                     "if parent_dir not in sys.path:\n"
                     "    sys.path.insert(0, parent_dir)\n"
        )
        if use_vision_tools:
            init_code += "from tools import *\n"
        
        init_resp = self.execute(init_code)
        print(init_resp[1])


    def cleanup(self):
        self.server.stop()
