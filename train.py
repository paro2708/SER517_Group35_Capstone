from GazeRefineNet import GRN
import pytorch_lightning as pl

if __name__ == '__main__':
    # image_dir = r'C:\\Rushi\\ProDataset\\train\\images\\iPhone 5S\\cropped_eyes'
    
    image_dir = r'C:\Users\rpatil29\ProDataset\train\iPhone 5S\cropped_eyes'
    meta_dir = r'C:\Users\rpatil29\ProDataset\train\meta'
    save_dir = "../Result"
 
    model = GRN(data_path= image_dir, save_path= save_dir)

    trainer = pl.Trainer(accelerator= "cpu", max_epochs= 1, default_root_dir= save_dir, enable_progress_bar= True)
    trainer.fit(model)

    print("Done")