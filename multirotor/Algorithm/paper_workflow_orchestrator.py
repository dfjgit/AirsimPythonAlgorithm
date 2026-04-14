from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from paper_workflow_archive import archive_comparison_stage_outputs, collect_ddpg_stage_outputs, collect_dqn_stage_outputs
from paper_workflow_recommendation import recommend_comparison_stage02
from paper_workflow_state import create_experiment_root, initialize_workflow_state, load_workflow_state, save_workflow_state


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

    def create_or_resume_experiment(self, *, workflow_type: str, alias: str = "") -> Path:
        exp_root = create_experiment_root(base_root=self.workflow_root, workflow_type=workflow_type, alias=alias)
        initialize_workflow_state(exp_root, workflow_type=workflow_type, alias=alias)
        return exp_root

    def load_state(self, exp_root: Path) -> dict:
        return load_workflow_state(exp_root)

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
                self.archive_runner(self.workspace_root, exp_root, **archive_kwargs)
            except Exception as exc:
                self._fail_step(exp_root, phase, exc)
                raise
        self._mark_step(exp_root, phase, "completed")

    def run_comparison_workflow(self, exp_root: Path) -> None:
        self._run_stage(
            exp_root,
            "stage01_ddpg",
            ["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Real_Environment.bat"],
            {"algorithm": "ddpg_apf", "stage_bucket": "stage01"},
        )
        self._run_stage(
            exp_root,
            "stage01_dqn",
            ["cmd.exe", "/d", "/c", "scripts\\Train_DQN_Movement_Real_Environment.bat"],
            {"algorithm": "pure_dqn", "stage_bucket": "stage01"},
        )
        self._run_stage(
            exp_root,
            "frozen_benchmark",
            ["cmd.exe", "/d", "/c", "scripts\\Run_Four_Group_Benchmark.bat"],
        )
        self._run_stage(
            exp_root,
            "training_comparison",
            ["python", "multirotor\\Algorithm\\visualize_training_data.py", "--auto", "--out", "analysis_results"],
        )
        self._mark_step(exp_root, "stage02_decision", "running")
        try:
            recommendations = self.recommendation_runner()
        except Exception as exc:
            self._fail_step(exp_root, "stage02_decision", exc)
            raise
        state = load_workflow_state(exp_root)
        state["recommendations"] = recommendations
        state["current_phase"] = "stage02_decision"
        state["status"] = "completed"
        state.setdefault("steps", {})["stage02_decision"] = {"status": "completed"}
        save_workflow_state(exp_root, state)

    def run_virtual_real_two_stage_workflow(self, exp_root: Path, *, refine_mode: str) -> None:
        self._run_stage(
            exp_root,
            "sim_pretrain",
            ["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Real_Environment.bat"],
            {"phase_bucket": "sim_pretrain", "refine_mode": ""},
        )

        refine_commands = {
            "online": ["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Crazyflie_Online_Single_Episode.bat"],
            "offline_logs": ["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Crazyflie_Logs.bat"],
        }
        if refine_mode not in refine_commands:
            raise ValueError(f"Unsupported refine_mode: {refine_mode}")

        self._run_stage(
            exp_root,
            "real_weighted_refine",
            refine_commands[refine_mode],
            {"phase_bucket": "real_weighted_refine", "refine_mode": refine_mode},
        )

        self._mark_step(exp_root, "two_stage_analysis", "running")
        try:
            analysis_outputs = self.two_stage_analysis_runner(exp_root, refine_mode=refine_mode)
        except Exception as exc:
            self._fail_step(exp_root, "two_stage_analysis", exc)
            raise
        self._mark_step(exp_root, "two_stage_analysis", "completed")

        self._mark_step(exp_root, "real_weighted_refine_decision", "running")
        try:
            recommendation = self.two_stage_recommendation_runner(analysis_outputs["summary_csv"])
        except Exception as exc:
            self._fail_step(exp_root, "real_weighted_refine_decision", exc)
            raise
        state = load_workflow_state(exp_root)
        state["recommendations"] = recommendation
        state["current_phase"] = "real_weighted_refine_decision"
        state["status"] = "completed"
        state.setdefault("steps", {})["real_weighted_refine_decision"] = {"status": "completed"}
        save_workflow_state(exp_root, state)


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


def create_default_orchestrator(*, workspace_root: Path) -> PaperWorkflowOrchestrator:
    workspace_root = Path(workspace_root)
    return PaperWorkflowOrchestrator(
        workspace_root=workspace_root,
        command_runner=_default_command_runner,
        archive_runner=archive_comparison_stage_outputs,
        recommendation_runner=lambda: _default_recommendation_runner(workspace_root),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper workflow orchestration tasks.")
    parser.add_argument(
        "--workflow",
        choices=["comparison"],
        required=True,
        help="Workflow to run. Comparison executes the stage01 comparison stack.",
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
    return parser


def main(argv: list[str] | None = None, *, orchestrator_factory=create_default_orchestrator) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace_root = Path(args.workspace_root).resolve()
    orchestrator = orchestrator_factory(workspace_root=workspace_root)
    exp_root = orchestrator.create_or_resume_experiment(workflow_type=args.workflow, alias=args.alias)

    if args.workflow == "comparison":
        print(f"[paper-workflow] comparison experiment: {exp_root}")
        orchestrator.run_comparison_workflow(exp_root)
        print(f"[paper-workflow] completed: {exp_root}")
        return 0

    parser.error(f"Unsupported workflow: {args.workflow}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
