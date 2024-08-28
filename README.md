# Image Encryption Using Base54 Matrix Transforms

## Overview

This project provides an encryption and decryption scheme for images using Base54 matrix transformations. The main functionality involves resizing images, applying matrix transformations for encryption, and reversing the transformations for decryption.

### Main Functions

- **`convertintobase54(img)`**  
  Resizes the input image to ensure its dimensions are multiples of 54x54 pixels.

- **`generate_random_moves()`**  
  Generates a random sequence of moves to be used for encryption and decryption.

- **`generateT()`**  
  Generates a set of 54x54 permutation matrices used for shuffling image blocks during encryption and decryption.

### Helper Functions

- **`encrypt_image(img, moves)`**  
  Applies the generated permutation matrices to the resized image based on the sequence of moves to produce the encrypted image.

- **`decrypt_image(img, moves)`**  
  Reverses the encryption process by applying the permutation matrices in reverse order, restoring the original image.

## How to Use

1. **Prepare the Image**: Resize the image using `convertintobase54()`.
2. **Encrypt the Image**: Use `encrypt_image()` with the resized image and a sequence of moves.
3. **Decrypt the Image**: Use `decrypt_image()` with the encrypted image and the same sequence of moves to recover the original image.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- MATLAB Documentation
- Image Processing Toolbox
