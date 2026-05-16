import os
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import requests
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import tkinter.font as tkfont

from huggingface_hub import HfApi

# --- Windows HiDPI ---
if sys.platform == "win32":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


@dataclass
class RepoItem:
    item_type: str   # "dir" / "file"
    path: str
    size: int = 0


class DownloadManager:
    def __init__(self):
        self.stop_flag = False
        self.is_paused = False

    def reset(self):
        self.stop_flag = False
        self.is_paused = False

    @staticmethod
    def format_size(size: int) -> str:
        if not size or size <= 0:
            return "-"
        v = float(size)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if v < 1024.0:
                return f"{v:.2f} {unit}"
            v /= 1024.0
        return f"{v:.2f} PB"

    @staticmethod
    def build_resolve_url(repo_id: str, repo_type: str, revision: str, file_path: str) -> str:
        """
        支持 model / dataset 两种仓库的 resolve 下载地址拼接
        model:   https://huggingface.co/<repo_id>/resolve/<rev>/<path>?download=true
        dataset: https://huggingface.co/datasets/<repo_id>/resolve/<rev>/<path>?download=true
        """
        fp = file_path.replace("\\", "/")
        if repo_type == "dataset":
            return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{fp}?download=true"
        return f"https://huggingface.co/{repo_id}/resolve/{revision}/{fp}?download=true"

    def download_single_file(
        self,
        session: requests.Session,
        model_id: str,
        file_path: str,
        save_dir: str,
        revision: str = "main",
        token: Optional[str] = None,
        expected_size: int = 0,
        progress_callback=None,
        status_callback=None,
        repo_type: str = "model",  # NEW: 支持 dataset
    ) -> Tuple[bool, str, int]:
        save_path = os.path.join(save_dir, file_path)
        os.makedirs(os.path.dirname(save_path) or save_dir, exist_ok=True)

        # skip if already exists (best-effort)
        if os.path.exists(save_path):
            local_size = os.path.getsize(save_path)
            if (expected_size and local_size == expected_size) or (not expected_size and local_size > 0):
                if status_callback:
                    status_callback(f"已存在：{self.format_size(local_size)}")
                return True, "文件已存在，跳过", local_size
            try:
                os.remove(save_path)
            except Exception:
                pass

        # NEW: 依据 repo_type 拼接 URL
        url = self.build_resolve_url(model_id, repo_type, revision, file_path)

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            if status_callback:
                status_callback("正在连接...")
            with session.get(url, stream=True, timeout=30, headers=headers) as resp:
                if resp.status_code in (401, 403):
                    return False, f"权限不足（{resp.status_code}）。私有/门控仓库请填写 Token。", 0
                resp.raise_for_status()

                total_size = int(resp.headers.get("content-length") or (expected_size or 0))
                downloaded = 0

                if status_callback:
                    if total_size > 0:
                        status_callback(f"开始下载：{os.path.basename(file_path)}（{self.format_size(total_size)}）")
                    else:
                        status_callback("开始下载...")

                last_ui = 0.0
                last_p = -1.0

                with open(save_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=256 * 1024):
                        while self.is_paused and not self.stop_flag:
                            time.sleep(0.2)
                        if self.stop_flag:
                            try:
                                f.close()
                            except Exception:
                                pass
                            try:
                                if os.path.exists(save_path):
                                    os.remove(save_path)
                            except Exception:
                                pass
                            return False, "下载已取消", 0

                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size > 0:
                            p = downloaded * 100.0 / total_size
                            now = time.time()
                            if (now - last_ui) >= 0.15 or (p - last_p) >= 1.0:
                                last_ui = now
                                last_p = p
                                progress_callback(p)

                final_size = os.path.getsize(save_path)
                if total_size > 0 and final_size != total_size:
                    try:
                        os.remove(save_path)
                    except Exception:
                        pass
                    return False, f"文件大小不匹配（实际:{final_size}，预期:{total_size}）", 0

                if progress_callback and total_size > 0:
                    progress_callback(100.0)
                return True, "下载成功", final_size

        except requests.exceptions.RequestException as e:
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
            except Exception:
                pass
            return False, f"网络错误：{e}", 0
        except Exception as e:
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
            except Exception:
                pass
            return False, f"错误：{e}", 0


class HuggingFaceAPI:
    def __init__(self):
        self.api = HfApi()

    # NEW: 自动识别仓库类型（dataset / model）
    def detect_repo_type(self, repo_id: str, token: Optional[str] = None) -> str:
        """
        规则：
        - dataset_info 能查到 => dataset
        - 否则尝试 model_info 能查到 => model
        - 两者都失败 => 默认 model（后续 list_repo_tree 也会失败，返回空）
        """
        try:
            self.api.dataset_info(repo_id, token=token)
            return "dataset"
        except Exception:
            pass

        try:
            self.api.model_info(repo_id, token=token)
            return "model"
        except Exception:
            return "model"

    # NEW: 返回 (items, repo_type)
    def get_repo_tree(
        self,
        model_id: str,
        revision: str = "main",
        token: Optional[str] = None
    ) -> Tuple[List[RepoItem], str]:
        repo_type = self.detect_repo_type(model_id, token=token)

        try:
            items = list(self.api.list_repo_tree(
                repo_id=model_id,
                repo_type=repo_type,  # NEW: 不再写死 model
                revision=revision,
                recursive=True,
                token=token
            ))
        except Exception:
            return [], repo_type

        results: List[RepoItem] = []
        for it in items:
            if hasattr(it, "size"):
                results.append(RepoItem("file", it.path, int(it.size or 0)))
            else:
                results.append(RepoItem("dir", it.path, 0))
        results.sort(key=lambda x: (0 if x.item_type == "dir" else 1, x.path.lower()))
        return results, repo_type


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hugging Face Downloader")

        self._apply_scaling_and_fonts()
        self._build_layout()

        self.api = HuggingFaceAPI()
        self.downloader = DownloadManager()
        self.session = requests.Session()

        self.repo_items: List[RepoItem] = []
        self.current_model_id = ""
        self.current_repo_type = "model"  # NEW: 当前仓库类型
        self.is_downloading = False
        self.total_files = 0
        self.success_count = 0
        self.failed_files: List[Tuple[str, str]] = []

        # --- window size: compute AFTER layout, then set minsize to guarantee status bar visible ---
        self.root.update_idletasks()
        req_w = self.root.winfo_reqwidth()
        req_h = self.root.winfo_reqheight()

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        # default window: large enough but not huge
        w = min(max(req_w, int(sw * 0.62)), 1400)
        h = min(max(req_h, int(sh * 0.78)), 980)

        self.root.geometry(f"{w}x{h}")

        # critical: minsize must be >= required size, otherwise bottom status may be clipped
        self.root.minsize(req_w, req_h)

        self.status_var.set("就绪")


    def _apply_scaling_and_fonts(self):
        # DPI-aware scaling (and slightly bump)
        scale = 1.35
        if sys.platform == "win32":
            try:
                from ctypes import windll
                dpi = windll.user32.GetDpiForSystem()
                scale = max(1.35, min(2.0, dpi / 96.0))
                scale = min(2.0, scale * 1.08)  # small bump (user asked a bit larger)
            except Exception:
                scale = 1.45
        try:
            self.root.tk.call("tk", "scaling", scale)
        except Exception:
            pass

        base_size = 14  # bigger than before
        fixed_size = 12

        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                f = tkfont.nametofont(name)
                f.configure(family="Microsoft YaHei UI", size=base_size)
            except Exception:
                pass
        try:
            tkfont.nametofont("TkFixedFont").configure(family="Consolas", size=fixed_size)
        except Exception:
            pass

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TButton", padding=(10, 5))
        style.configure("Small.TButton", padding=(9, 4))
        style.configure("TLabelframe", padding=8)
        style.configure("TLabelframe.Label", font=("Microsoft YaHei UI", base_size, "bold"))

        style.configure("Treeview", font=("Microsoft YaHei UI", base_size), rowheight=38)
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", base_size, "bold"))
        style.configure("green.Horizontal.TProgressbar", background='green')

    # ============================
    # UI 布局：完全保持原样（未改）
    # ============================
    def _build_layout(self):
        # Use GRID for the whole app so the status bar is always visible.
        self.root.grid_rowconfigure(0, weight=1)   # 主内容区随窗口扩展
        self.root.grid_rowconfigure(1, weight=0)   # 状态栏固定
        self.root.grid_columnconfigure(0, weight=1)

        self.container = ttk.Frame(self.root, padding=8)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.container.grid_rowconfigure(3, weight=1)  # content_frame expands
        self.container.grid_columnconfigure(0, weight=1)

        # Row0: model settings
        top = ttk.LabelFrame(self.container, text="模型设置")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.grid_columnconfigure(1, weight=1)

        ttk.Label(top, text="模型/数据集ID：").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        self.model_id_var = tk.StringVar(value="Manojb/stable-diffusion-2-1-base")
        self.model_id_entry = ttk.Entry(top, textvariable=self.model_id_var, width=55)
        self.model_id_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=6)

        ttk.Label(top, text="revision：").grid(row=0, column=2, sticky="w", padx=(10, 8), pady=6)
        self.revision_var = tk.StringVar(value="main")
        self.revision_entry = ttk.Entry(top, textvariable=self.revision_var, width=14)
        self.revision_entry.grid(row=0, column=3, sticky="w", padx=(0, 8), pady=6)

        self.load_btn = ttk.Button(top, text="查询", command=self.load_files)
        self.load_btn.grid(row=0, column=4, sticky="w", pady=6)

        ttk.Label(top, text="Token(可选)：").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        self.token_var = tk.StringVar(value="")
        self.token_entry = ttk.Entry(top, textvariable=self.token_var, show="•")
        self.token_entry.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=6)
        ttk.Label(top, text="私有/门控模型需要 Token").grid(row=1, column=2, columnspan=3, sticky="w", pady=6)

        # Row1: save settings
        pathf = ttk.LabelFrame(self.container, text="保存设置")
        pathf.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        pathf.grid_columnconfigure(1, weight=1)

        ttk.Label(pathf, text="保存路径：").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        self.save_path_var = tk.StringVar(value=os.path.abspath("./downloaded_models"))
        self.save_path_entry = ttk.Entry(pathf, textvariable=self.save_path_var)
        self.save_path_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=6)
        ttk.Button(pathf, text="浏览…", command=self.browse_path).grid(row=0, column=2, sticky="w", pady=6)
        ttk.Button(pathf, text="打开目录", command=self.open_save_dir, style="Small.TButton").grid(row=0, column=3, sticky="w", padx=(8, 0), pady=6)

        # Row2: controls + progress (two frames side-by-side)
        row2 = ttk.Frame(self.container)
        row2.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        row2.grid_columnconfigure(0, weight=4)
        row2.grid_columnconfigure(1, weight=1)

        ctrl = ttk.LabelFrame(row2, text="下载控制")
        ctrl.grid(row=0, column=1, sticky="nsew")

        button_container = ttk.Frame(ctrl)
        button_container.pack(expand=True, fill="both", padx=10, pady=6)
        btn_frame = ttk.Frame(button_container)
        btn_frame.pack(expand=True)  # 居中显示

        self.start_btn = ttk.Button(btn_frame, text="开始下载", command=self.start_download, state="disabled")
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.pause_btn = ttk.Button(btn_frame, text="暂停", command=self.toggle_pause, state="disabled")
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self.stop_download, state="disabled")
        self.stop_btn.pack(side=tk.LEFT)

        prog = ttk.LabelFrame(row2, text="下载进度")
        prog.grid(row=0, column=0, sticky="nsew",padx=(0, 20))
        prog.grid_columnconfigure(1, weight=2)
        prog.grid_columnconfigure(2, weight=0)

        ttk.Label(prog, text="总体：").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        self.overall_pb = ttk.Progressbar(prog, orient="horizontal", mode="determinate", maximum=100,style="green.Horizontal.TProgressbar")
        self.overall_pb.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=6)
        self.overall_lab = ttk.Label(prog, text="0%")
        self.overall_lab.grid(row=0, column=2, sticky="e", padx=(0, 8), pady=6)

        ttk.Label(prog, text="当前：").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        self.file_pb = ttk.Progressbar(prog, orient="horizontal", mode="determinate", maximum=100,style="green.Horizontal.TProgressbar")
        self.file_pb.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=6)
        self.file_lab = ttk.Label(prog, text="等待开始")
        self.file_lab.grid(row=1, column=2, sticky="e", padx=(0, 8), pady=6)

        # Row3: content (file list + log). IMPORTANT: log has fixed height and does NOT expand.
        content = ttk.Frame(self.container)
        content.grid(row=3, column=0, sticky="nsew")
        content.grid_rowconfigure(0, weight=1)  # file list expands
        content.grid_rowconfigure(1, weight=0)  # log fixed
        content.grid_columnconfigure(0, weight=1)

        list_frame = ttk.LabelFrame(content, text="仓库树")
        list_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        columns = ("选择", "类型", "路径", "大小", "状态")
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        col_cfg = [
            ("选择", 70, "center"),
            ("类型", 80, "center"),
            ("路径", 600, "w"),
            ("大小", 120, "center"),
            ("状态", 130, "center"),
        ]
        for c, w, anchor in col_cfg:
            self.file_tree.heading(c, text=c)
            self.file_tree.column(c, width=w, anchor=anchor, stretch=(c == "路径"))

        ybar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_tree.yview)
        xbar = ttk.Scrollbar(list_frame, orient="horizontal", command=self.file_tree.xview)
        self.file_tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)

        self.file_tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")

        self.file_tree.bind("<Button-1>", self._on_tree_click)

        toolbar = ttk.Frame(list_frame)
        toolbar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.select_all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="全选", variable=self.select_all_var, command=self.toggle_select_all).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="反选", command=self.invert_selection, style="Small.TButton").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(toolbar, text="仅勾选大文件(>100MB)", command=self.select_large_files, style="Small.TButton").pack(side=tk.LEFT, padx=(10, 0))

        log_frame = ttk.LabelFrame(content, text="下载日志")
        log_frame.grid(row=1, column=0, sticky="ew")
        log_frame.grid_columnconfigure(0, weight=1)

        # fixed-height log: do NOT make it expand vertically
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, font=("Consolas", 12))
        self.log_text.grid(row=0, column=0, sticky="ew")

        log_btns = ttk.Frame(log_frame)
        log_btns.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(log_btns, text="清空日志", command=self.clear_log, style="Small.TButton").pack(side=tk.LEFT)
        ttk.Button(log_btns, text="复制日志", command=self.copy_log, style="Small.TButton").pack(side=tk.LEFT, padx=(10, 0))

        # Row4: status bar (ALWAYS visible)
        self.status_var = tk.StringVar(value="")
        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 2))

    # --- UI helpers ---
    def ui(self, func, *args, **kwargs):
        self.root.after(0, lambda: func(*args, **kwargs))

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")

        def _append():
            self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
            self.log_text.see(tk.END)

        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.ui(_append)

    def _set_progress(self, pb: ttk.Progressbar, value: float):
        pb.configure(value=max(0.0, min(100.0, float(value))))

    # --- File tree selection ---
    def _on_tree_click(self, event):
        region = self.file_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        col = self.file_tree.identify_column(event.x)
        if col != "#1":
            return
        row = self.file_tree.identify_row(event.y)
        if not row:
            return
        vals = list(self.file_tree.item(row, "values"))
        if vals[1] == "dir":
            return
        vals[0] = "✗" if vals[0] == "✓" else "✓"
        self.file_tree.item(row, values=vals)
        self.select_all_var.set(self._all_files_checked())

    def _all_files_checked(self) -> bool:
        for item in self.file_tree.get_children():
            vals = self.file_tree.item(item, "values")
            if vals and vals[1] == "file" and vals[0] != "✓":
                return False
        return True

    def toggle_select_all(self):
        check = self.select_all_var.get()
        for item in self.file_tree.get_children():
            vals = list(self.file_tree.item(item, "values"))
            if vals[1] == "file":
                vals[0] = "✓" if check else "✗"
                self.file_tree.item(item, values=vals)

    def invert_selection(self):
        for item in self.file_tree.get_children():
            vals = list(self.file_tree.item(item, "values"))
            if vals[1] == "file":
                vals[0] = "✗" if vals[0] == "✓" else "✓"
                self.file_tree.item(item, values=vals)
        self.select_all_var.set(self._all_files_checked())

    def select_large_files(self):
        for item in self.file_tree.get_children():
            vals = list(self.file_tree.item(item, "values"))
            if vals[1] != "file":
                continue
            size_str = vals[3]
            should = False
            try:
                if "GB" in size_str:
                    should = True
                elif "MB" in size_str:
                    mb = float(size_str.split()[0])
                    should = mb >= 100
            except Exception:
                should = False
            vals[0] = "✓" if should else "✗"
            self.file_tree.item(item, values=vals)
        self.select_all_var.set(self._all_files_checked())

    # --- Common buttons ---
    def browse_path(self):
        path = filedialog.askdirectory()
        if path:
            self.save_path_var.set(path)
            self.log(f"保存路径：{path}")

    def open_save_dir(self):
        try:
            os.startfile(os.path.abspath(self.save_path_var.get()))
        except Exception:
            messagebox.showinfo("提示", "无法打开目录（可能路径不存在）")

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def copy_log(self):
        txt = self.log_text.get("1.0", tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        self.status_var.set("已复制日志到剪贴板")

    # --- Load repo tree ---
    def load_files(self):
        model_id = self.model_id_var.get().strip()
        revision = self.revision_var.get().strip() or "main"
        token = self.token_var.get().strip() or None

        if not model_id:
            messagebox.showwarning("警告", "请输入模型ID")
            return

        self.load_btn.config(state="disabled")
        self.start_btn.config(state="disabled")
        self.status_var.set("正在加载仓库树...")
        self.log(f"加载仓库树：{model_id} ({revision}) ...")

        threading.Thread(target=self._load_files_worker, args=(model_id, revision, token), daemon=True).start()

    def _load_files_worker(self, model_id: str, revision: str, token: Optional[str]):
        # NEW: 返回 items + repo_type
        items, repo_type = self.api.get_repo_tree(model_id, revision=revision, token=token)
        self.current_model_id = model_id
        self.current_repo_type = repo_type  # NEW: 记录仓库类型
        self.repo_items = items

        def refresh():
            self.file_tree.delete(*self.file_tree.get_children())
            for it in items:
                if it.item_type != "file":
                    continue
                self.file_tree.insert("", "end", values=("✓", "file", it.path, DownloadManager.format_size(it.size), "等待下载"))
            self.select_all_var.set(True)

            file_count = sum(1 for i in items if i.item_type == "file")
            dir_count = sum(1 for i in items if i.item_type == "dir")
            # NEW: 显示类型信息（不改布局，仅改文案）
            self.status_var.set(f"已加载：类型 {repo_type}，目录 {dir_count}，文件 {file_count}")
            self.load_btn.config(state="normal")
            self.start_btn.config(state="normal" if file_count > 0 else "disabled")

        if not items:
            self.log("⚠️ 未获取到仓库树（可能：仓库不存在 / 需要 Token / 网络问题）。")
        self.ui(refresh)

    # --- Download ---
    def start_download(self):
        if self.is_downloading:
            return

        revision = self.revision_var.get().strip() or "main"
        token = self.token_var.get().strip() or None
        save_dir = self.save_path_var.get().strip()
        if not save_dir:
            messagebox.showwarning("警告", "请选择保存路径")
            return
        os.makedirs(save_dir, exist_ok=True)

        checked = set()
        for row in self.file_tree.get_children():
            vals = self.file_tree.item(row, "values")
            if vals and vals[1] == "file" and vals[0] == "✓":
                checked.add(vals[2])

        files = [it for it in self.repo_items if it.item_type == "file" and it.path in checked]
        if not files:
            messagebox.showwarning("警告", "请选择要下载的文件（目录无法下载）")
            return

        self.downloader.reset()
        self.is_downloading = True
        self.total_files = len(files)
        self.success_count = 0
        self.failed_files = []

        self.start_btn.config(state="disabled")
        self.pause_btn.config(state="normal", text="暂停")
        self.stop_btn.config(state="normal")
        self.load_btn.config(state="disabled")

        self._set_progress(self.overall_pb, 0)
        self.overall_lab.config(text="0%")
        self._set_progress(self.file_pb, 0)
        self.file_lab.config(text="等待开始")

        self.status_var.set("下载中...")
        # NEW: 记录类型信息，便于排查
        self.log(
            f"开始下载：{self.current_model_id}（类型 {self.current_repo_type}），共 {len(files)} 个文件。保存到：{save_dir}"
        )

        threading.Thread(target=self._download_worker, args=(files, save_dir, revision, token), daemon=True).start()

    def _mark_status_by_path(self, file_path: str, status: str):
        for row in self.file_tree.get_children():
            vals = self.file_tree.item(row, "values")
            if vals and vals[1] == "file" and vals[2] == file_path:
                new_vals = list(vals)
                new_vals[4] = status
                self.file_tree.item(row, values=new_vals)
                return

    def _download_worker(self, files: List[RepoItem], save_dir: str, revision: str, token: Optional[str]):
        try:
            for idx, it in enumerate(files, start=1):
                if self.downloader.stop_flag:
                    break

                fp = it.path
                self.ui(self._mark_status_by_path, fp, "下载中")
                self.ui(self.file_lab.config, text="正在下载...")
                self.ui(self._set_progress, self.file_pb, 0)

                self.log(f"[{idx}/{len(files)}] {fp}")

                # 创建自定义状态回调
                def status_handler(status: str):
                    if "开始下载：" in status:
                        # 只显示"正在下载..."，不显示文件名
                        self.ui(self.file_lab.config, text="正在下载...")
                    elif "已存在：" in status:
                        self.ui(self.file_lab.config, text="文件已存在")
                    else:
                        # 其他状态直接显示
                        self.ui(self.file_lab.config, text=status)

                ok, msg, size = self.downloader.download_single_file(
                    session=self.session,
                    model_id=self.current_model_id,
                    file_path=fp,
                    save_dir=save_dir,
                    revision=revision,
                    token=token,
                    expected_size=it.size,
                    repo_type=self.current_repo_type,  # NEW: 关键参数
                    progress_callback=lambda p: self.ui(self._set_progress, self.file_pb, p),
                    status_callback=status_handler,
                )

                if ok:
                    self.success_count += 1
                    self.ui(self._mark_status_by_path, fp, "完成")
                    self.log(f"✅ {msg}（{DownloadManager.format_size(size)}）")
                    self.ui(self.file_lab.config, text="下载完成")
                else:
                    self.failed_files.append((fp, msg))
                    self.ui(self._mark_status_by_path, fp, "失败")
                    self.log(f"❌ {msg}")
                    self.ui(self.file_lab.config, text="下载失败")

                overall = idx * 100.0 / len(files)
                self.ui(self._set_progress, self.overall_pb, overall)
                self.ui(self.overall_lab.config, text=f"{overall:.1f}%")

        except Exception as e:
            self.log(f"❌ 下载线程异常：{e}")
        finally:
            self.ui(self._download_finished)

    def toggle_pause(self):
        if not self.is_downloading:
            return
        if self.downloader.is_paused:
            self.downloader.is_paused = False
            self.pause_btn.config(text="暂停")
            self.status_var.set("下载中...")
            self.log("继续下载")
        else:
            self.downloader.is_paused = True
            self.pause_btn.config(text="继续")
            self.status_var.set("已暂停")
            self.log("暂停下载")

    def stop_download(self):
        if not self.is_downloading:
            return
        self.downloader.stop_flag = True
        self.status_var.set("正在停止...")
        self.log("用户请求停止下载（将取消当前文件）")

    def _download_finished(self):
        self.is_downloading = False
        self.start_btn.config(state="normal" if any(i.item_type == "file" for i in self.repo_items) else "disabled")
        self.pause_btn.config(state="disabled", text="暂停")
        self.stop_btn.config(state="disabled")
        self.load_btn.config(state="normal")

        self._set_progress(self.file_pb, 0)
        self.file_lab.config(text="下载完成" if not self.downloader.stop_flag else "已停止")

        if not self.downloader.stop_flag:
            self._set_progress(self.overall_pb, 100)
            self.overall_lab.config(text="100%")

        self.log("=" * 60)
        self.log(f"下载结束：成功 {self.success_count}/{self.total_files}，失败 {len(self.failed_files)}")
        if self.failed_files:
            self.log("失败列表：")
            for fp, reason in self.failed_files:
                self.log(f"  - {fp}: {reason}")

        self.status_var.set("完成" if not self.downloader.stop_flag else "已停止")

        if not self.downloader.stop_flag:
            if messagebox.askyesno("完成", "下载完成！是否打开保存目录？"):
                self.open_save_dir()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
