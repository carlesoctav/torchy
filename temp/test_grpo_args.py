from trl import GRPOConfig, GRPOTrainer


args = GRPOConfig(use_cpu=True)
GRPOTrainer()
print("DEBUGPRINT {args}:", args)
