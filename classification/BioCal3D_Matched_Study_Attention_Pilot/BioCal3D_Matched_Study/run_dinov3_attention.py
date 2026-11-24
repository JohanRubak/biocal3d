"""Compare learned attention and the matched logistic baseline on frozen DINOv3 tokens."""

from attention_pilot import main


if __name__ == "__main__":
    main("dinov3")
