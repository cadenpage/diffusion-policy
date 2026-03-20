set shell := ["bash", "-euo", "pipefail", "-c"]

repo_root := justfile_directory()
dataset_root := repo_root + "/data"
rotom_repo_id := "local/rotom_task_teleop"
rotom_dataset_dir := dataset_root + "/local/rotom_task_teleop"
rotom_rsync_source := "Berra:~/Documents/rotom/data/lerobot/local/rotom_task_teleop/"
rotom_policy_path := repo_root + "/outputs/train/rotom_task_teleop_diffusion/checkpoints/last/pretrained_model"
venv_dir := repo_root + "/.venv"
venv_python := venv_dir + "/bin/python"
venv_pip := venv_dir + "/bin/pip"
venv_jupyter := venv_dir + "/bin/jupyter"

default:
    @just --list

venv-create python="python3.12":
    py='{{python}}'; py="${py#python=}"; "$py" -m venv "{{venv_dir}}"

venv-install:
    test -x "{{venv_python}}"
    "{{venv_python}}" -m pip install --upgrade pip setuptools wheel
    "{{venv_pip}}" install -r requirements.txt

notebook:
    test -x "{{venv_jupyter}}"
    "{{venv_jupyter}}" lab scripts/rotom_lerobot.ipynb

sync-rotom-dataset source="{{rotom_rsync_source}}" destination="{{rotom_dataset_dir}}/":
    mkdir -p "{{dataset_root}}/local"
    rsync -av --progress "{{source}}" "{{destination}}"

check-rotom-dataset repo_id="{{rotom_repo_id}}" root="{{dataset_root}}":
    test -d "{{root}}/{{repo_id}}"
    test -x "{{venv_python}}"
    "{{venv_python}}" scripts/rotom_lerobot.py verify-dataset --repo-id "{{repo_id}}" --root "{{root}}"

infer-rotom-checkpoint repo_id="{{rotom_repo_id}}" root="{{dataset_root}}" checkpoint="{{rotom_policy_path}}" device="mps" index="0":
    test -x "{{venv_python}}"
    "{{venv_python}}" scripts/rotom_lerobot.py infer-checkpoint \
      --repo-id "{{repo_id}}" \
      --root "{{root}}" \
      --policy "{{checkpoint}}" \
      --device "{{device}}" \
      --index {{index}}
