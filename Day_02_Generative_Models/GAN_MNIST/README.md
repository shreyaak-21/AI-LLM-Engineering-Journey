Generative Adversarial Networks on MNIST
Vanilla GAN & Conditional GAN (Keras / TensorFlow)

This repository contains a practical implementation of Generative Adversarial Networks (GANs) using the MNIST handwritten digits dataset.
It includes both:

Vanilla GAN – learns to generate handwritten digits from noise

Conditional GAN (cGAN) – generates specific digits based on class labels (0–9)

The project focuses on hands-on understanding of generative models, not academic theory.

📌 Project Objectives

Understand how GANs work in practice

Implement Generator and Discriminator using Keras

Observe training dynamics of GANs

Learn how Conditional GANs solve limitations of vanilla GANs

Build a GitHub-ready generative model project


⚙️ Technologies Used

Python 3.9+

TensorFlow / Keras

NumPy

Matplotlib

MNIST Dataset

🔍 Vanilla GAN (train_gan.py)
How it works

Generator

Takes random noise as input

Generates fake MNIST-like images

Discriminator

Classifies images as real or fake

Both models are trained adversarially

Key Limitation

No control over which digit is generated

Model may generate any digit randomly

Output Example

Generated digits gradually improve over epochs:

outputs/
├── epoch_5.png
├── epoch_15.png
├── epoch_30.png
└── epoch_50.png

🎯 Conditional GAN (train_cgan.py)
Why Conditional GAN?

Vanilla GANs cannot control output class.
Conditional GANs solve this by conditioning generation on labels.

How it works

Generator input: Noise + digit label (0–9)

Discriminator input: Image + digit label

Model learns:

“Generate digit X when label X is provided”

Benefits

Controlled image generation

Reduced mode collapse

More structured outputs

Output Example

Each saved image shows digits 0–9 generated intentionally:

outputs_cgan/
├── epoch_10.png
├── epoch_25.png
└── epoch_50.png

▶️ How to Run the Project
1️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train Vanilla GAN
python train_gan.py

4️⃣ Train Conditional GAN
python train_cgan.py

⏱ Training Details

Dataset: MNIST (28×28 grayscale)

Epochs: 50

Batch size: 64

Runs on CPU (no GPU required)

Training time: ~10–15 minutes per model (CPU)

🧠 Key Learnings

Practical understanding of adversarial training

Generator vs Discriminator dynamics

Why mode collapse occurs

How Conditional GANs improve controllability

End-to-end generative model workflow