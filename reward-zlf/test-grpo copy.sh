# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

# 缓解显存碎片（对大 seq_len 场景经常有帮助）
# 注意：expandable_segments 与 vLLM 的 memory pool（cumem allocator）冲突，会导致 EngineCore 启动失败
# 如需缓解碎片，建议后续改为 PYTORCH_ALLOC_CONF（或关闭 vLLM memory pool）再评估
# export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "[debug] reward_manager.source=importlib"
echo "[debug] reward_manager.name=NaturalFeedbackRewardManager"
echo "[debug] reward_manager.module.path=/nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/reward/natural_feedback_reward.py"

echo "[debug] reward_model.reward_loop_source=importlib"
echo "[debug] reward_model.reward_loop_class_name=NaturalFeedbackRewardLoopManager"
echo "[debug] reward_model.reward_loop_module_path=/nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/reward/natural_feedback_reward.py"

# Make sure the custom reward manager module can be imported by workers.
export PYTHONPATH=/nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf:${PYTHONPATH}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/data/train.json \
    data.val_files=/nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/data/test.json \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    trainer.default_local_dir=/nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/checkpoints/testv1 \
    actor_rollout_ref.model.path=/nfsdata-117/model/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=6e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_manager.source=importlib \
    reward_manager.name=NaturalFeedbackRewardManager \
    reward_manager.module.path=/nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/reward/natural_feedback_reward.py \
    reward_model.reward_loop_source=importlib \
    reward_model.reward_loop_class_name=NaturalFeedbackRewardLoopManager \
    reward_model.reward_loop_module_path=/nfsdata-117/project/finvlr1/zlf/guotai-reward/verl/reward-zlf/reward/natural_feedback_reward.py \
    +reward_model.reward_kwargs.enable_process_reward=True \
    +reward_model.reward_kwargs.process_reward_lambda=0.5 \
    +reward_model.reward_kwargs.process_similarity_threshold=0.2 \
    trainer.critic_warmup=0.05 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='guotai-verl' \
    trainer.experiment_name='test-grpo-4b-ins-5k' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@
