a = im2double(imread('boom.jpg'));
a = convertintobase54(a);
moves1 = generate_random_moves();
moves2 = generate_random_moves();
moves3 = generate_random_moves();
moves4 = generate_random_moves();

encrypted_img = imencrypt(a,moves1,moves2,moves3,moves4);
encrypted_img = imencrypt(permute(encrypted_img, [2 1 3]),moves1,moves2,moves3,moves4);
%encrypted_img = imencrypt(permute(encrypted_img, [2 1 3]),moves1,moves2,moves3);
%encrypted_img = imencrypt(permute(encrypted_img, [2 1 3]),moves1,moves2,moves3);
%imshow(encrypted_img);
imwrite(encrypted_img, 'encrypted_image_2.jpg');
decrypted_img = imdecrypt(encrypted_img,moves1,moves2,moves3,moves4);
decrypted_img = imdecrypt(permute(decrypted_img, [2 1 3]),moves1,moves2,moves3,moves4);
%decrypted_img = imdecrypt(permute(decrypted_img, [2 1 3]),moves1,moves2,moves3);
%decrypted_img = imdecrypt(permute(decrypted_img, [2 1 3]),moves1,moves2,moves3);
% 
imwrite(decrypted_img,'decrypted_image_1.jpg')
subplot(1, 3, 1); 
imshow(a);
title('Original Image');
subplot(1, 3, 2); 
imshow(encrypted_img);
title('encrypted Image');
subplot(1, 3, 3); 
imshow(decrypted_img);
title('decrypted Image');
% figure 
% subplot(1, 3, 1); 
% imhist(a);
% title('Original Image');
% subplot(1, 3, 2); 
% imhist(encrypted_img);
% title('encrypted Image');
% subplot(1, 3, 3); 
% imhist(decrypted_img);
% title('decrypted Image');
% histogram(encrypted_img)
% entropy_encrypted = cal_entropy(encrypted_img);
% disp(entropy_encrypted)
% s = size(encrypted_img);
% random_img = zeros(s);
% for i = 1:s(1)
%     for j = 1:s(2)
%         for k = 1:s(3)
%             random_img(i,j,k) = rand();
%         end
%     end
% end
% figure
% imshow(random_img)
% entropy_random = cal_entropy(random_img);
% disp(entropy_random)
% 
% entropy_decrypted = cal_entropy(decrypted_img);
% disp(entropy_decrypted)
%b = 1:54;
%T = generateT();
%move1 = make_shuffle_mat(T, moves1);
% shuffle = b*move1;
% disp(shuffle);
% move1 = inv(move1);
% b =  move1 \ shuffle';
% disp(b)

function out_img = imencrypt(a, moves1, moves2, moves3, moves4)
    T = generateT();

    move1 = make_shuffle_mat(T, moves1);
    move2 = make_shuffle_mat(T, moves2);
    move3 = make_shuffle_mat(T, moves3);
    move4 = make_shuffle_mat(T, moves4);

    R = a(:,:,1);
    G = a(:,:,2);
    B = a(:,:,3);
    shuffledR = shuffle_channel(R,move1,move4);
    shuffledG = shuffle_channel(G,move2,move4);
    shuffledB = shuffle_channel(B,move3,move4);
    out_img = cat(3, shuffledR, shuffledG, shuffledB);
end
function shuffledR = shuffle_channel(R,move1,move4)
    [rows, cols] = size(R);
    I = shuffleI(rows,move4*move1);
    J = shuffleI(cols,move4*move1);
    p = 1:54:rows;
    q = 1:54:cols;

    shuffledR = zeros(rows, cols);
    
    for i = 1:length(I)
        for j = 1:length(J)
            blockR = R(I(i):I(i)+53, J(j):J(j)+53);            
            shuffledR(p(i):p(i)+53, q(j):q(j)+53) = blockR * move1;
            progress = ((i-1)*length(J) + j )*100/(length(I)*length(J)) ;
            disp(progress);
        end
    end
end
function out_img = imdecrypt(a, moves1, moves2, moves3, moves4)
    T = generateT();
    
    move1 = make_shuffle_mat(T, moves1);
    move2 = make_shuffle_mat(T, moves2);
    move3 = make_shuffle_mat(T, moves3);
    move4 = make_shuffle_mat(T, moves4);

    R = a(:,:,1);
    G = a(:,:,2);
    B = a(:,:,3);
    shuffledR = unshuffle(R,move1,move4);
    shuffledG = unshuffle(G,move2,move4);
    shuffledB = unshuffle(B,move3,move4);
    out_img = cat(3, shuffledR, shuffledG, shuffledB);
end
function shuffledR = unshuffle(R,move1,move4)
    [rows, cols] = size(R);
    p = 1:54:rows;
    q = 1:54:cols;
    I = shuffleI(rows, move4*move1);
    I = p * generatePermutationMatrix(I,p);
    J = shuffleI(cols, move4*move1);
    J = q * generatePermutationMatrix(J,q);

    shuffledR = zeros(rows, cols);
    move1 = inv(move1);
    for i = 1:length(I)
        for j = 1:length(J)
            blockR = R(I(i):I(i)+53, J(j):J(j)+53);
            shuffledR(p(i):p(i)+53, q(j):q(j)+53) = move1 \ blockR;
            progress = ((i-1)*length(J) + j )*100/(length(I)*length(J));
            disp(progress);
        end
    end
end
function I_reconstructed = shuffleI(cols,move4)
    I = 1:54:cols;
    remainder = mod(length(I), 54);
    if remainder ~= 0
        numZerosToAdd = 54 - remainder;
        I = [I, zeros(1, numZerosToAdd)];
    end
    subArrays = reshape(I, 54, [])';
    
    for i = 1:size(subArrays, 1)
        subArrays(i, :) = subArrays(i, :) * move4;
    end
    subArrays(subArrays == 0) = [];
    I_reconstructed = subArrays';
    I_reconstructed = I_reconstructed(:)';
end

function T = generatePermutationMatrix(original_array, shuffled_array)

    original_array = original_array(:);
    shuffled_array = shuffled_array(:);
    
    if length(original_array) ~= length(shuffled_array)
        error('Input arrays must have the same length.');
    end
    
    n = length(original_array);
    
    T = zeros(n, n);
    
    for i = 1:n
        pos = find(original_array == shuffled_array(i));
        T(pos, i) = 1;
    end
end
function move = make_shuffle_mat(T, moves)
    move = eye(54);
    
    for i = 1:length(moves)
        move = move * T{moves(i)};
    end
end
function moves = generate_random_moves()
    moves = randi([1, 12], 1, 20);
end

function T = generateT()
    cube = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53];

    moves = {[6, 3, 0, 7, 4, 1, 8, 5, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 21, 22, 46, 24, 25, 47, 42, 28, 29, 43, 31, 32, 44, 34, 35, 36, 37, 38, 39, 40, 41, 26, 23, 20, 33, 30, 27, 48, 49, 50, 51, 52, 53]
    [2, 5, 8, 1, 4, 7, 0, 3, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 44, 21, 22, 43, 24, 25, 42, 47, 28, 29, 46, 31, 32, 45, 34, 35, 36, 37, 38, 39, 40, 41, 27, 30, 33, 20, 23, 26, 48, 49, 50, 51, 52, 53]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 12, 9, 16, 13, 10, 17, 14, 11, 38, 19, 20, 37, 22, 23, 36, 25, 26, 27, 28, 53, 30, 31, 52, 33, 34, 51, 29, 32, 35, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 18, 21, 24]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 17, 10, 13, 16, 9, 12, 15, 51, 19, 20, 52, 22, 23, 53, 25, 26, 27, 28, 36, 30, 31, 37, 33, 34, 38, 24, 21, 18, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 35, 32, 29]
    [36, 1, 2, 39, 4, 5, 42, 7, 8, 9, 10, 51, 12, 13, 48, 15, 16, 45, 24, 21, 18, 25, 22, 19, 26, 23, 20, 27, 28, 29, 30, 31, 32, 33, 34, 35, 17, 37, 38, 14, 40, 41, 11, 43, 44, 0, 46, 47, 3, 49, 50, 6, 52, 53]
    [45, 1, 2, 48, 4, 5, 51, 7, 8, 9, 10, 42, 12, 13, 39, 15, 16, 36, 20, 23, 26, 19, 22, 25, 18, 21, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 37, 38, 3, 40, 41, 6, 43, 44, 17, 46, 47, 14, 49, 50, 11, 52, 53]
    [0, 1, 47, 3, 4, 50, 6, 7, 53, 44, 10, 11, 41, 13, 14, 38, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 33, 30, 27, 34, 31, 28, 35, 32, 29, 36, 37, 2, 39, 40, 5, 42, 43, 8, 45, 46, 15, 48, 49, 12, 51, 52, 9]
    [0, 1, 38, 3, 4, 41, 6, 7, 44, 53, 10, 11, 50, 13, 14, 47, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 32, 35, 28, 31, 34, 27, 30, 33, 36, 37, 15, 39, 40, 12, 42, 43, 9, 45, 46, 2, 48, 49, 5, 51, 52, 8]
    [27, 28, 29, 3, 4, 5, 6, 7, 8, 18, 19, 20, 12, 13, 14, 15, 16, 17, 0, 1, 2, 21, 22, 23, 24, 25, 26, 9, 10, 11, 30, 31, 32, 33, 34, 35, 42, 39, 36, 43, 40, 37, 44, 41, 38, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    [18, 19, 20, 3, 4, 5, 6, 7, 8, 27, 28, 29, 12, 13, 14, 15, 16, 17, 9, 10, 11, 21, 22, 23, 24, 25, 26, 0, 1, 2, 30, 31, 32, 33, 34, 35, 38, 41, 44, 37, 40, 43, 36, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    [0, 1, 2, 3, 4, 5, 24, 25, 26, 9, 10, 11, 12, 13, 14, 33, 34, 35, 18, 19, 20, 21, 22, 23, 15, 16, 17, 27, 28, 29, 30, 31, 32, 6, 7, 8, 36, 37, 38, 39, 40, 41, 42, 43, 44, 51, 48, 45, 52, 49, 46, 53, 50, 47]
    [0, 1, 2, 3, 4, 5, 33, 34, 35, 9, 10, 11, 12, 13, 14, 24, 25, 26, 18, 19, 20, 21, 22, 23, 6, 7, 8, 27, 28, 29, 30, 31, 32, 15, 16, 17, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 50, 53, 46, 49, 52, 45, 48, 51]
    };
    T = cell(1, length(moves));
    for i = 1:length(moves)
        T{i} = generatePermutationMatrix(cube, moves{i});
    end
end
function resized_img = convertintobase54(img)

    %img = imresize(img, 0.5);
    [rows, cols, ~] = size(img);

    new_rows = round(rows / 54) * 54;
    new_cols = round(cols / 54) * 54;
    resized_img = imresize(img, [new_rows, new_cols]);
    
end
function Entropy = cal_entropy(image)
    Entropy_R = entropy(image(:,:,1));
    Entropy_G = entropy(image(:,:,2));
    Entropy_B = entropy(image(:,:,3));
    Entropy = [Entropy_R,Entropy_G,Entropy_B];
end
