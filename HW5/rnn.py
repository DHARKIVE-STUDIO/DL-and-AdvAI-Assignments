import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
import numpy as np
import os
import matplotlib.pyplot as plt


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


# Tokenizer class for encoding and decoding captions
class Tokenizer:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}

        # Initialize with basic tokens and known vocabulary
        vocab = ["<START>", "<EOS>", "This", "is", "number"] + [str(i) for i in range(10)]
        for idx, word in enumerate(vocab):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

    def __len__(self):
        vocab_size = len(self.word_to_idx)
        return vocab_size

    def encode(self, caption):
        return (
            [self.word_to_idx["<START>"]]
            + [self.word_to_idx[word] for word in caption.split()]
            + [self.word_to_idx["<EOS>"]]
        )

    def decode(self, tokens):
        words = [
            self.idx_to_word[token] for token in tokens if token in self.idx_to_word
        ]
        return " ".join(words).replace("<EOS>", "").replace("<START>", "").strip()


class ImageCaptioningRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=32):
        set_seed()
        super(ImageCaptioningRNN, self).__init__()
        self.vocab_size = vocab_size

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        self.fc = nn.Linear(32 * 5 * 5, hidden_dim)  # CNN output: 32 channels * 5x5

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions):
        flatten_features = self.cnn(images)
        hidden_init = self.fc(flatten_features).unsqueeze(0)

        captions_embedded = self.embedding(captions)
        rnn_output, _ = self.rnn(captions_embedded, hidden_init)
        outputs = self.fc_out(rnn_output)
        return outputs


def generate_caption_labels(tokenizer, labels):
    captions = [tokenizer.encode(f"This is number {label.item()}") for label in labels]

    number_of_tokens = max(len(c) for c in captions)
    batch_size = len(captions)

    captions_label = torch.zeros(batch_size, number_of_tokens, dtype=torch.long)
    for i, caption in enumerate(captions):
        captions_label[i, :len(caption)] = torch.tensor(caption)

    return captions_label


def train(model, tokenizer, mnist_subset, num_epochs=2000, batch_size=100, only_train_one_batch=False):
    set_seed()
    data_loader = DataLoader(mnist_subset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in data_loader:
            captions_label = generate_caption_labels(tokenizer=tokenizer, labels=labels)
            
            model_outputs = model(images, captions_label[:, :-1])  # input without <EOS>

            outputs = model_outputs.reshape(-1, model.vocab_size)
            targets = captions_label[:, 1:].reshape(-1)  # target without <START>

            loss = criterion(outputs, targets)

            if only_train_one_batch:
                return loss, outputs, targets

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"image_captioning_model_epoch.pth")
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Model Saved")


def evaluate_model(
    model,
    tokenizer,
    mnist_subset,
    num_of_eval_images=2,
    max_number_of_predicted_tokens=6,
):
    model.eval()
    with torch.no_grad():
        random_indices = random.sample(range(len(mnist_subset)), num_of_eval_images)
        images, labels = zip(*[mnist_subset[idx] for idx in random_indices])
        images = torch.stack(images)

        start_token_idx = torch.tensor(
            [tokenizer.word_to_idx["<START>"]], dtype=torch.long
        ).to(images.device)
        caption_input = (
            model.embedding(start_token_idx).unsqueeze(0).expand(len(images), -1, -1)
        )
        features = model.cnn(images)
        hidden_init = model.fc(features).unsqueeze(0)

        batch_predicted_tokens = [[] for _ in range(len(images))]

        for _ in range(max_number_of_predicted_tokens):
            outputs, hidden_init = model.rnn(caption_input, hidden_init)
            output_tokens = model.fc_out(outputs[:, -1, :]).argmax(dim=-1)

            for i, token in enumerate(output_tokens):
                batch_predicted_tokens[i].append(token.item())

            if all(token == tokenizer.word_to_idx["<EOS>"] for token in output_tokens):
                break

            caption_input = model.embedding(output_tokens).unsqueeze(1)

        plt.figure(figsize=(15, 5))
        font_size = 8 if num_of_eval_images <= 4 else 6

        for i, (predicted_tokens, image, label) in enumerate(
            zip(batch_predicted_tokens, images, labels)
        ):
            caption = tokenizer.decode(predicted_tokens)
            plt.subplot(1, num_of_eval_images, i + 1)
            plt.imshow(image.squeeze(0).cpu().numpy(), cmap="gray")
            plt.title(
                f"Predicted Caption: {caption}\nTrue Label: This is number {label}",
                fontsize=font_size,
            )
            plt.axis("off")

        plt.tight_layout()
        plt.show()