set shell := ["bash", "-euo", "pipefail", "-c"]

repo_root := justfile_directory()
dataset_root := repo_root + "/data"
rotom_repo_id := "local/rotom_task_teleop"
rotom_dataset_dir := dataset_root + "/local/rotom_task_teleop"
rotom_rsync_source := "Berra:~/Documents/rotom/data/lerobot/local/rotom_task_teleop/"
rotom_host := "Berra"
rotom_output_dir := repo_root + "/outputs/train/rotom_task_teleop_diffusion"
rotom_policy_path := rotom_output_dir + "/checkpoints/last/pretrained_model"
rotom_remote_policy_dir := "~/Documents/rotom/policies/rotom_task_teleop_diffusion/pretrained_model"
rotom_hub_dataset_repo := "caden-ut/rotom_task_teleop"
rotom_hub_policy_repo := "caden-ut/rotom_task_teleop_diffusion"
venv_dir := repo_root + "/.venv"
venv_python := venv_dir + "/bin/python"
venv_pip := venv_dir + "/bin/pip"
venv_hf := venv_dir + "/bin/hf"
venv_lerobot_train := venv_dir + "/bin/lerobot-train"

default:
    @just --list

venv-create python="python3.12":
    py='{{python}}'; py="${py#python=}"; "$py" -m venv "{{venv_dir}}"

venv-install:
    test -x "{{venv_python}}"
    "{{venv_python}}" -m pip install --upgrade pip setuptools wheel
    "{{venv_pip}}" install -r requirements.txt

hf-login:
    test -x "{{venv_hf}}"
    "{{venv_hf}}" auth login

hf-whoami:
    test -x "{{venv_hf}}"
    "{{venv_hf}}" auth whoami

sync-rotom-dataset:
    mkdir -p "{{dataset_root}}/local"
    rsync -av --progress "{{rotom_rsync_source}}" "{{rotom_dataset_dir}}/"

check-rotom-dataset:
    test -d "{{rotom_dataset_dir}}"
    test -x "{{venv_python}}"
    "{{venv_python}}" scripts/rotom_lerobot.py verify-dataset --repo-id "{{rotom_repo_id}}" --root "{{dataset_root}}"

train-rotom-diffusion device="mps" steps="5000" batch_size="8" num_workers="0":
    test -x "{{venv_lerobot_train}}"
    "{{venv_lerobot_train}}" \
      --dataset.repo_id="{{rotom_repo_id}}" \
      --dataset.root="{{dataset_root}}" \
      --policy.type=diffusion \
      --policy.device="{{device}}" \
      --policy.use_amp=false \
      --policy.n_obs_steps=2 \
      --batch_size={{batch_size}} \
      --num_workers={{num_workers}} \
      --steps={{steps}} \
      --save_freq=1000 \
      --log_freq=50 \
      --wandb.enable=false \
      --output_dir="{{rotom_output_dir}}" \
      --job_name=rotom_task_teleop_diffusion

push-rotom-dataset-hub dataset_repo_id="{{rotom_hub_dataset_repo}}" private="true":
    test -d "{{rotom_dataset_dir}}"
    test -x "{{venv_python}}"
    "{{venv_python}}" scripts/rotom_lerobot.py push-folder-to-hub \
      --repo-id "{{dataset_repo_id}}" \
      --folder "{{rotom_dataset_dir}}" \
      --repo-type dataset \
      --private "{{private}}" \
      --commit-message "Upload Rotom LeRobot dataset"

check-rotom-dataset-hub dataset_repo_id="{{rotom_hub_dataset_repo}}":
    test -x "{{venv_python}}"
    "{{venv_python}}" scripts/rotom_lerobot.py verify-dataset --repo-id "{{dataset_repo_id}}" --root ""

train-rotom-diffusion-hub dataset_repo_id="{{rotom_hub_dataset_repo}}" policy_repo_id="{{rotom_hub_policy_repo}}" device="cuda" steps="5000" batch_size="8" num_workers="2":
    test -x "{{venv_lerobot_train}}"
    "{{venv_lerobot_train}}" \
      --dataset.repo_id="{{dataset_repo_id}}" \
      --policy.type=diffusion \
      --policy.device="{{device}}" \
      --policy.use_amp=false \
      --policy.n_obs_steps=2 \
      --policy.repo_id="{{policy_repo_id}}" \
      --batch_size={{batch_size}} \
      --num_workers={{num_workers}} \
      --steps={{steps}} \
      --save_freq=1000 \
      --log_freq=50 \
      --wandb.enable=false \
      --output_dir="{{rotom_output_dir}}" \
      --job_name=rotom_task_teleop_diffusion

infer-rotom-diffusion-hub dataset_repo_id="{{rotom_hub_dataset_repo}}" policy_ref="{{rotom_hub_policy_repo}}" device="cpu" index="0":
    test -x "{{venv_python}}"
    "{{venv_python}}" scripts/rotom_lerobot.py infer-checkpoint \
      --repo-id "{{dataset_repo_id}}" \
      --policy "{{policy_ref}}" \
      --root "" \
      --device "{{device}}" \
      --index {{index}}

push-rotom-policy-hub policy_repo_id="{{rotom_hub_policy_repo}}" checkpoint="{{rotom_policy_path}}" private="true":
    test -d "{{checkpoint}}"
    test -x "{{venv_python}}"
    "{{venv_python}}" scripts/rotom_lerobot.py push-folder-to-hub \
      --repo-id "{{policy_repo_id}}" \
      --folder "{{checkpoint}}" \
      --repo-type model \
      --private "{{private}}" \
      --commit-message "Upload Rotom diffusion policy"

infer-rotom-diffusion checkpoint="{{rotom_policy_path}}" device="mps" index="0":
    test -x "{{venv_python}}"
    "{{venv_python}}" scripts/rotom_lerobot.py infer-checkpoint \
      --repo-id "{{rotom_repo_id}}" \
      --root "{{dataset_root}}" \
      --policy "{{checkpoint}}" \
      --device "{{device}}" \
      --index {{index}}

push-rotom-policy checkpoint="{{rotom_policy_path}}" destination="{{rotom_host}}:{{rotom_remote_policy_dir}}/":
    test -d "{{checkpoint}}"
    ssh "{{rotom_host}}" "mkdir -p {{rotom_remote_policy_dir}}"
    rsync -av --progress "{{checkpoint}}/" "{{destination}}"
