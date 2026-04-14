from __future__ import annotations

from pathlib import Path

from paper_workflow_state import create_experiment_root, initialize_workflow_state, load_workflow_state, save_workflow_state


class PaperWorkflowOrchestrator:
    def __init__(self, *, workspace_root: Path, command_runner, archive_runner, recommendation_runner):
        self.workspace_root = Path(workspace_root)
        self.workflow_root = self.workspace_root / "analysis_results" / "workflows"
        self.command_runner = command_runner
        self.archive_runner = archive_runner
        self.recommendation_runner = recommendation_runner

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
