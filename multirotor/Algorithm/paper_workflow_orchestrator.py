from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from paper_two_stage_analysis import build_two_stage_summary
from paper_two_stage_recommendation import recommend_real_weighted_continue
from paper_workflow_archive import (
    archive_comparison_stage_outputs,
    archive_two_stage_outputs,
    collect_ddpg_stage_outputs,
    collect_dqn_stage_outputs,
)
from paper_workflow_recommendation import recommend_comparison_stage02
from paper_workflow_state import (
    create_experiment_root,
    initialize_workflow_state,
    list_resumable_experiments,
    load_workflow_state,
    save_workflow_state,
)


def _localized_text(zh_text: str, en_text: str) -> str:
    return zh_text if os.environ.get("AIRSIM_UI_LANG", "").lower() == "zh" else en_text


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class PaperWorkflowOrchestrator:
    def __init__(
        self,
        *,
        workspace_root: Path,
        command_runner,
        archive_runner,
        recommendation_runner,
        two_stage_recommendation_runner=None,
        two_stage_analysis_runner=None,
    ):
        self.workspace_root = Path(workspace_root)
        self.workflow_root = self.workspace_root / "analysis_results" / "workflows"
        self.command_runner = command_runner
        self.archive_runner = archive_runner
        self.recommendation_runner = recommendation_runner
        self.two_stage_recommendation_runner = two_stage_recommendation_runner
        self.two_stage_analysis_runner = two_stage_analysis_runner

    def find_latest_resumable_experiment(self, *, workflow_type: str) -> dict | None:
        resumable = list_resumable_experiments(self.workflow_root, workflow_type=workflow_type)
        return resumable[0] if resumable else None

    def create_or_resume_experiment(
        self,
        *,
        workflow_type: str,
        alias: str = "",
        resume_latest: bool = False,
        experiment_root: Path | None = None,
    ) -> Path:
        if experiment_root is not None:
            return Path(experiment_root)
        if resume_latest:
            latest = self.find_latest_resumable_experiment(workflow_type=workflow_type)
            if latest is not None:
                return Path(latest["experiment_root"])
        exp_root = create_experiment_root(base_root=self.workflow_root, workflow_type=workflow_type, alias=alias)
        initialize_workflow_state(exp_root, workflow_type=workflow_type, alias=alias)
        return exp_root

    def load_state(self, exp_root: Path) -> dict:
        return load_workflow_state(exp_root)

    def _step_status(self, exp_root: Path, phase: str) -> str:
        state = load_workflow_state(exp_root)
        return str(state.get("steps", {}).get(phase, {}).get("status") or "")

    def _is_step_completed(self, exp_root: Path, phase: str) -> bool:
        return self._step_status(exp_root, phase) == "completed"

    def _mark_step(self, exp_root: Path, phase: str, status: str) -> dict:
        state = load_workflow_state(exp_root)
        state["current_phase"] = phase
        if status == "running" and state.get("status") != "failed":
            state["status"] = "running"
        state.setdefault("steps", {})[phase] = {"status": status}
        save_workflow_state(exp_root, state)
        return state

    def _fail_step(self, exp_root: Path, phase: str, reason: str | None = None) -> dict:
        state = load_workflow_state(exp_root)
        state["current_phase"] = phase
        state["status"] = "failed"
        step_state = state.setdefault("steps", {}).setdefault(phase, {})
        step_state["status"] = "failed"
        if reason:
            step_state["error"] = str(reason)
        save_workflow_state(exp_root, state)
        return state

    def _complete_step(self, exp_root: Path, phase: str, *, recommendations=None) -> dict:
        state = load_workflow_state(exp_root)
        state["current_phase"] = phase
        state["status"] = "completed"
        if recommendations is not None:
            state["recommendations"] = recommendations
        state.setdefault("steps", {})[phase] = {"status": "completed"}
        save_workflow_state(exp_root, state)
        return state

    def _run_stage(self, exp_root: Path, phase: str, command: list[str], archive_kwargs: dict | None = None) -> None:
        self._mark_step(exp_root, phase, "running")
        try:
            exit_code = self.command_runner(command, cwd=self.workspace_root)
        except Exception as exc:
            self._fail_step(exp_root, phase, exc)
            raise
        if exit_code != 0:
            self._fail_step(exp_root, phase, f"Command exited with {exit_code}")
            raise RuntimeError(f"Command {command} failed with exit code {exit_code}")
        if archive_kwargs:
            try:
                archive_outputs = self.archive_runner(self.workspace_root, exp_root, **archive_kwargs)
            except Exception as exc:
                self._fail_step(exp_root, phase, exc)
                raise
            state = load_workflow_state(exp_root)
            state.setdefault("artifacts", {})[phase] = _json_safe(archive_outputs)
            save_workflow_state(exp_root, state)
        self._mark_step(exp_root, phase, "completed")

    def run_comparison_workflow(self, exp_root: Path) -> None:
        apf_output_root = exp_root / "artifacts" / "apf_baseline_sim"
        if not self._is_step_completed(exp_root, "apf_baseline_sim"):
            self._run_stage(
                exp_root,
                "apf_baseline_sim",
                [
                    "cmd.exe",
                    "/d",
                    "/c",
                    "scripts\\Run_APF_Baseline_Simulation.bat",
                    "--out",
                    str(apf_output_root),
                    "--raw-log-dir",
                    str(apf_output_root / "logs"),
                    "--experiment-id",
                    exp_root.name,
                    "--stage-name",
                    "stage00_apf_baseline",
                    "--stage-index",
                    "0",
                ],
            )
        if not self._is_step_completed(exp_root, "stage01_ddpg"):
            self._run_stage(
                exp_root,
                "stage01_ddpg",
                ["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Real_Environment.bat"],
                {"algorithm": "ddpg_apf", "stage_bucket": "stage01"},
            )
        if not self._is_step_completed(exp_root, "stage01_dqn"):
            self._run_stage(
                exp_root,
                "stage01_dqn",
                ["cmd.exe", "/d", "/c", "scripts\\Train_DQN_Movement_Real_Environment.bat"],
                {"algorithm": "pure_dqn", "stage_bucket": "stage01"},
            )
        if not self._is_step_completed(exp_root, "frozen_benchmark"):
            self._run_stage(
                exp_root,
                "frozen_benchmark",
                ["cmd.exe", "/d", "/c", "scripts\\Run_Four_Group_Benchmark.bat"],
            )
        if not self._is_step_completed(exp_root, "training_comparison"):
            self._run_stage(
                exp_root,
                "training_comparison",
                ["python", "multirotor\\Algorithm\\visualize_training_data.py", "--auto", "--out", "analysis_results"],
            )
        if self._is_step_completed(exp_root, "stage02_decision"):
            self._complete_step(exp_root, "stage02_decision")
            return
        self._mark_step(exp_root, "stage02_decision", "running")
        try:
            recommendations = self.recommendation_runner()
        except Exception as exc:
            self._fail_step(exp_root, "stage02_decision", exc)
            raise
        self._complete_step(exp_root, "stage02_decision", recommendations=recommendations)

    def run_virtual_real_two_stage_workflow(self, exp_root: Path, *, refine_mode: str) -> None:
        refine_commands = {
            "online": ["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Crazyflie_Online_Single_Episode.bat"],
            "offline_logs": ["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Crazyflie_Logs.bat"],
        }
        if refine_mode not in refine_commands:
            self._fail_step(exp_root, "real_weighted_refine", f"Unsupported refine_mode: {refine_mode}")
            raise ValueError(f"Unsupported refine_mode: {refine_mode}")

        if not self._is_step_completed(exp_root, "sim_pretrain"):
            self._run_stage(
                exp_root,
                "sim_pretrain",
                ["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Real_Environment.bat"],
                {"phase_bucket": "sim_pretrain", "refine_mode": ""},
            )

        if not self._is_step_completed(exp_root, "real_weighted_refine"):
            self._run_stage(
                exp_root,
                "real_weighted_refine",
                refine_commands[refine_mode],
                {"phase_bucket": "real_weighted_refine", "refine_mode": refine_mode},
            )

        analysis_outputs = None
        if self._is_step_completed(exp_root, "two_stage_analysis"):
            state = load_workflow_state(exp_root)
            analysis_outputs = state.get("artifacts", {}).get("two_stage_analysis", {})
        else:
            self._mark_step(exp_root, "two_stage_analysis", "running")
            try:
                analysis_outputs = self.two_stage_analysis_runner(exp_root, refine_mode=refine_mode)
            except Exception as exc:
                self._fail_step(exp_root, "two_stage_analysis", exc)
                raise
            state = load_workflow_state(exp_root)
            state.setdefault("artifacts", {})["two_stage_analysis"] = _json_safe(analysis_outputs)
            save_workflow_state(exp_root, state)
            self._mark_step(exp_root, "two_stage_analysis", "completed")

        if self._is_step_completed(exp_root, "real_weighted_refine_decision"):
            self._complete_step(exp_root, "real_weighted_refine_decision")
            return

        summary_csv = Path(analysis_outputs["summary_csv"])
        self._mark_step(exp_root, "real_weighted_refine_decision", "running")
        try:
            recommendation = self.two_stage_recommendation_runner(summary_csv)
        except Exception as exc:
            self._fail_step(exp_root, "real_weighted_refine_decision", exc)
            raise
        self._complete_step(exp_root, "real_weighted_refine_decision", recommendations=recommendation)


def _repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_command_runner(command: list[str], *, cwd: Path) -> int:
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return int(completed.returncode)


def _select_training_log(logs: list[Path], prefix: str) -> Path | None:
    for log_path in logs:
        if log_path.name.startswith(prefix):
            return log_path
    return logs[0] if logs else None


def _default_recommendation_runner(workspace_root: Path) -> dict:
    benchmark_csv = workspace_root / "analysis_results" / "four_group_benchmark" / "four_group_eval_episodes.csv"
    if not benchmark_csv.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {benchmark_csv}")

    ddpg_outputs = collect_ddpg_stage_outputs(workspace_root, stage_name="stage01")
    dqn_outputs = collect_dqn_stage_outputs(workspace_root, stage_name="stage01")
    ddpg_training_csv = _select_training_log(ddpg_outputs.get("training_logs", []), "ddpg_training_")
    dqn_training_csv = _select_training_log(dqn_outputs.get("training_logs", []), "dqn_training_")
    if ddpg_training_csv is None:
        raise FileNotFoundError("No DDPG training CSV found for stage01 recommendation")
    if dqn_training_csv is None:
        raise FileNotFoundError("No DQN training CSV found for stage01 recommendation")

    return {
        "ddpg_apf": recommend_comparison_stage02(
            ddpg_training_csv,
            benchmark_csv,
            algorithm_type="ddpg_apf",
        ),
        "pure_dqn": recommend_comparison_stage02(
            dqn_training_csv,
            benchmark_csv,
            algorithm_type="pure_dqn",
        ),
    }


def _sorted_logs(log_root: Path, pattern: str) -> list[Path]:
    return sorted(log_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def _default_two_stage_analysis_runner(workspace_root: Path, exp_root: Path, *, refine_mode: str) -> dict[str, Path]:
    sim_logs = _sorted_logs(exp_root / "artifacts" / "sim_pretrain" / "ddpg_apf" / "logs", "*.csv")
    sim_training_csv = _select_training_log(sim_logs, "ddpg_training_")
    if sim_training_csv is None:
        sim_training_csv = _select_training_log(
            collect_ddpg_stage_outputs(workspace_root, stage_name="stage01").get("training_logs", []),
            "ddpg_training_",
        )
    if sim_training_csv is None:
        raise FileNotFoundError("No DDPG stage01 training CSV found for two-stage summary")

    refine_patterns = {
        "online": "crazyflie_training_online*.csv",
        "offline_logs": "crazyflie_training_logs*.csv",
    }
    refine_prefixes = {
        "online": "crazyflie_training_online",
        "offline_logs": "crazyflie_training_logs",
    }
    refine_logs = _sorted_logs(
        exp_root / "artifacts" / "real_weighted_refine" / refine_mode / "logs",
        "*.csv",
    )
    refine_training_csv = _select_training_log(refine_logs, refine_prefixes[refine_mode])
    if refine_training_csv is None:
        refine_training_csv = _select_training_log(
            _sorted_logs(
                workspace_root / "multirotor" / "DDPG_Weight" / "crazyflie_logs",
                refine_patterns[refine_mode],
            ),
            refine_prefixes[refine_mode],
        )
    if refine_training_csv is None:
        raise FileNotFoundError(f"No refine training CSV found for two-stage summary ({refine_mode})")

    return build_two_stage_summary(
        sim_training_csv,
        refine_training_csv,
        exp_root / "analysis" / "two_stage",
    )


def _default_archive_runner(workspace_root: Path, exp_root: Path, **archive_kwargs) -> dict:
    if "algorithm" in archive_kwargs:
        return archive_comparison_stage_outputs(workspace_root, exp_root, **archive_kwargs)
    return archive_two_stage_outputs(workspace_root, exp_root, **archive_kwargs)


def create_default_orchestrator(*, workspace_root: Path) -> PaperWorkflowOrchestrator:
    workspace_root = Path(workspace_root)
    return PaperWorkflowOrchestrator(
        workspace_root=workspace_root,
        command_runner=_default_command_runner,
        archive_runner=_default_archive_runner,
        recommendation_runner=lambda: _default_recommendation_runner(workspace_root),
        two_stage_recommendation_runner=recommend_real_weighted_continue,
        two_stage_analysis_runner=lambda exp_root, refine_mode: _default_two_stage_analysis_runner(
            workspace_root,
            exp_root,
            refine_mode=refine_mode,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper workflow orchestration tasks.")
    parser.add_argument(
        "--workflow",
        choices=["comparison", "virtual_real_two_stage"],
        required=True,
        help="Workflow to run. Comparison executes the stage01 comparison stack; virtual_real_two_stage runs sim pretrain plus real refine.",
    )
    parser.add_argument(
        "--refine-mode",
        choices=["online", "offline_logs"],
        default="online",
        help="Real refine mode for the virtual_real_two_stage workflow. Defaults to online.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=_repo_root_from_module(),
        help="Repository workspace root. Defaults to the current repository containing this module.",
    )
    parser.add_argument(
        "--alias",
        default="",
        help="Optional experiment alias used in the workflow experiment directory name.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the latest unfinished experiment for the selected workflow instead of creating a new one.",
    )
    parser.add_argument(
        "--query-latest-resumable",
        action="store_true",
        help="Print the latest unfinished experiment metadata for the selected workflow and exit.",
    )
    return parser


def main(argv: list[str] | None = None, *, orchestrator_factory=create_default_orchestrator) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace_root = Path(args.workspace_root).resolve()
    orchestrator = orchestrator_factory(workspace_root=workspace_root)
    if args.query_latest_resumable:
        latest = orchestrator.find_latest_resumable_experiment(workflow_type=args.workflow)
        if latest is not None:
            state = latest["state"]
            print(f"{latest['experiment_root']}|{state.get('status', '')}|{state.get('current_phase', '')}")
        return 0

    exp_root = orchestrator.create_or_resume_experiment(
        workflow_type=args.workflow,
        alias=args.alias,
        resume_latest=bool(args.resume_latest),
    )

    if args.workflow == "comparison":
        print(_localized_text(f"[paper-workflow] 四组统一仿真对比阶段已启动: {exp_root}", f"[paper-workflow] comparison experiment: {exp_root}"))
        orchestrator.run_comparison_workflow(exp_root)
        print(_localized_text(f"[paper-workflow] 四组统一仿真对比阶段已完成: {exp_root}", f"[paper-workflow] completed: {exp_root}"))
        return 0

    if args.workflow == "virtual_real_two_stage":
        print(
            _localized_text(
                f"[paper-workflow] 虚实两阶段实验已启动: {exp_root} (refine_mode={args.refine_mode})",
                f"[paper-workflow] virtual-real two-stage experiment: {exp_root} (refine_mode={args.refine_mode})",
            )
        )
        orchestrator.run_virtual_real_two_stage_workflow(exp_root, refine_mode=args.refine_mode)
        print(_localized_text(f"[paper-workflow] 虚实两阶段实验已完成: {exp_root}", f"[paper-workflow] completed: {exp_root}"))
        return 0

    parser.error(f"Unsupported workflow: {args.workflow}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
