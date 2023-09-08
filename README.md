# SeismicRepGAN

This code allows to train and test the `RepGAN` model [1] conceived in [2] to predict the transient dynamic response of a multi-storey building, 
considering different damage conditions (simulated as stiffness reduction of a random storey) under earthquake strong ground motion.
The code allows to classify the input time histories (one per storey) within 3 different damage classes (stiffness reduction of 0%, 25%, 50%).
Moreover, the `RepGAN` can effectively perform domain translation, for instance by encoding the undamaged structural response into the latent
clustered manifold and decode it into the transient reponse under damage conditions (by simply switching the corresponding cluster before hand).

Full paper: https://linkinghub.elsevier.com/retrieve/pii/S026772612300386X

## References
[1] Zhou, Y.; Gu, K.; Huang, T. Unsupervised Representation Adversarial Learning Network: From Reconstruction to Generation. 
arXiv April 6, 2019. http://arxiv.org/abs/1804.07353 (accessed 2023-04-01).

[2] Gatti, F.; Rosafalco, L.; Colombera, G.; Mariani, S.; Corigliano, A. Multi-Storey Shear Type Buildings under Earthquake Loading:
Adversarial Learning-Based Prediction of the Transient Dynamics and Damage Classification. Soil Dynamics and Earthquake Engineering 2023,
173, 108141. https://doi.org/10.1016/j.soildyn.2023.108141.
