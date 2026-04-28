clear; clc;

addpath(genpath('C:\Users\user\Desktop\AIwireless\cost2100-master'));

out_dir = 'CsiNetData';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

dataset_names = {
    'user_center'
    'user_edge'
    'user_uniform'
    'user_left_cluster'
    'user_right_cluster'
    'user_ring'
};

num_train = 3000;
num_val   = 600;
num_test  = 600;

img_height = 32;
img_width  = 32;

for d = 1:length(dataset_names)
    dataset_name = dataset_names{d};

    fprintf('Generating dataset: %s\n', dataset_name);

    HT_train = generate_one_dataset(dataset_name, num_train, img_height, img_width);
    HT_val   = generate_one_dataset(dataset_name, num_val,   img_height, img_width);
    HT_test  = generate_one_dataset(dataset_name, num_test,  img_height, img_width);

    HT = single(HT_train);
    save(fullfile(out_dir, ['DATA_Htrainin_' dataset_name '.mat']), 'HT', '-v7.3');

    HT = single(HT_val);
    save(fullfile(out_dir, ['DATA_Hvalin_' dataset_name '.mat']), 'HT', '-v7.3');

    HT = single(HT_test);
    save(fullfile(out_dir, ['DATA_Htestin_' dataset_name '.mat']), 'HT', '-v7.3');
end

fprintf('Done. Files are saved in CsiNetData folder.\n');


function HT = generate_one_dataset(dataset_name, num_samples, img_height, img_width)

    HT = zeros(num_samples, 2 * img_height * img_width, 'single');

    for n = 1:num_samples

        user_pos = sample_user_position(dataset_name);

        % TODO:
        % 正式版這裡要換成 COST2100 產生的通道矩陣。
        % 目前這行只是先測試資料格式是否正確。
        H_complex = randn(img_height, img_width) + 1j * randn(img_height, img_width);

        H_real = normalize_to_01(real(H_complex));
        H_imag = normalize_to_01(imag(H_complex));

        H_2ch = zeros(2, img_height, img_width);
        H_2ch(1,:,:) = H_real;
        H_2ch(2,:,:) = H_imag;

        HT(n, :) = reshape(H_2ch, 1, []);
    end
end


function pos = sample_user_position(dataset_name)

    switch dataset_name
        case 'user_center'
            x = -20 + 40 * rand();
            y = -20 + 40 * rand();

        case 'user_edge'
            r = 80 + 20 * rand();
            theta = 2*pi*rand();
            x = r*cos(theta);
            y = r*sin(theta);

        case 'user_uniform'
            x = -100 + 200 * rand();
            y = -100 + 200 * rand();

        case 'user_left_cluster'
            x = -80 + 20 * randn();
            y = 0 + 20 * randn();

        case 'user_right_cluster'
            x = 80 + 20 * randn();
            y = 0 + 20 * randn();

        case 'user_ring'
            r = 50 + 10 * randn();
            theta = 2*pi*rand();
            x = r*cos(theta);
            y = r*sin(theta);

        otherwise
            error('Unknown dataset name.');
    end

    z = 1.5;
    pos = [x; y; z];
end


function X = normalize_to_01(X)
    X = X - min(X(:));
    X = X ./ (max(X(:)) + eps);
end