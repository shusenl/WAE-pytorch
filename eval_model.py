from trainer_gan import *

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
trainer = Trainer(args)
checkpointFileName = 20000
trainer.load_checkpoint(checkpointFileName)
