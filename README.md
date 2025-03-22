# Chest X-Ray Disease Classifier

This project is part of the 2025 [UBC Medicine Datathon](https://datascienceandhealth.ubc.ca/events/ubc-medicine-datathon), where we attempt to predict and classify
diseases given X-ray images from the [NIH Chest X-rays Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data/data?select=train_val_list.txt) on Kaggle.

Our approach to this problem involved fine-tuning [CheXNet](https://github.com/arnoweng/CheXNet), a pre-trained CNN based on the popular DenseNet CNN model.
We fine-tune our model over **five specific diseases** that are particularily difficult to identify, according to [this NIH report](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf)

- Nodules
- Masses
- Pneumonia
- Pneumothorax
- Atelectasis

## Contributors
Agam S, Timothy S, Fazeeia M, Benjamin F, Steven C, Shirsha G.

## References

- Mu, Y (2017). CheXNet for Classification and Localization of Thoracic Diseases. GitHub. https://github.com/arnoweng/CheXNet

- Pranav Rajpurkar, Jeremy Irvin, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Curtis Langlotz, Katie Shpanskaya, Matthew P. Lungren, & Andrew Y. Ng. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.

- Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR 2017,