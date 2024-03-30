from openGaze import openGaze
import pytorch_lightning as pl

if __name__ == '__main__':
    dataset_dir = '../ProDataset/'
    save_dir = '../Result/'

    model = openGaze(data_path= dataset_dir, save_path= save_dir)

    trainer = pl.Trainer(accelerator= "cpu", max_epochs= 10, default_root_dir= save_dir, enable_progress_bar= True)
    trainer.fit(model)

    print("Done")