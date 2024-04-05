from GazeRefineNet import GRN
import pytorch_lightning as pl

if __name__ == '__main__':
    
    # image_dir = r'C:\Users\rpatil29\ProDataset\train\iPhone 5S\cropped_eyes'
    # meta_dir = r'C:\Users\rpatil29\ProDataset\train\meta'
    # save_dir = "../Result"

    dataset_dir = '../ProDataset/'
    save_dir = '../Result/'
 
    model = GRN(data_path= dataset_dir, save_path= save_dir)

    trainer = pl.Trainer(accelerator= "gpu", max_epochs= 5, default_root_dir= save_dir, enable_progress_bar= True)
    trainer.fit(model)

    print("Done")