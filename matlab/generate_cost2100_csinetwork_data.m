clear; clc; close all;

root_dir = 'C:\\Users\\user\\Desktop\\AIwireless';
cost2100_dir = fullfile(root_dir, 'cost2100-master');
addpath(genpath(cost2100_dir));

out_dir = fullfile(root_dir, 'CsiNetData');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

dataset_names = {'user_center'; 'user_edge'; 'user_uniform'; 'user_left_cluster'; 'user_right_cluster'; 'user_ring'};
num_train = 3000;
num_val = 600;
num_test = 600;
img_height = 32;
img_width = 32;

for d = 1:length(dataset_names)
    dataset_name = dataset_names{d};
    fprintf('Generating dataset: %s\n', dataset_name);
    HT_train = generate_one_split(dataset_name, num_train, img_height, img_width);
    HT_val = generate_one_split(dataset_name, num_val, img_height, img_width);
    HT_test = generate_one_split(dataset_name, num_test, img_height, img_width);

    HT = single(HT_train); save(fullfile(out_dir, ['DATA_Htrainin_' dataset_name '.mat']), 'HT', '-v7.3');
    HT = single(HT_val);   save(fullfile(out_dir, ['DATA_Hvalin_' dataset_name '.mat']), 'HT', '-v7.3');
    HT = single(HT_test);  save(fullfile(out_dir, ['DATA_Htestin_' dataset_name '.mat']), 'HT', '-v7.3');
end
fprintf('Done. Files are saved in CsiNetData folder.\n');

function HT = generate_one_split(dataset_name, num_samples, img_height, img_width)
    HT = zeros(num_samples, 2 * img_height * img_width, 'single');
    for n = 1:num_samples
        MSPos = sample_user_position(dataset_name);
        H_complex = generate_cost2100_channel(MSPos, img_height, img_width);
        H_real = real(H_complex); H_imag = imag(H_complex);
        scale = max([abs(H_real(:)); abs(H_imag(:)); 1e-12]);
        H_real_norm = min(max(H_real / (2 * scale) + 0.5, 0), 1);
        H_imag_norm = min(max(H_imag / (2 * scale) + 0.5, 0), 1);
        H_2ch = zeros(2, img_height, img_width);
        H_2ch(1,:,:) = H_real_norm; H_2ch(2,:,:) = H_imag_norm;
        HT(n,:) = reshape(H_2ch, 1, []);
    end
end

function H_complex = generate_cost2100_channel(MSPos, img_height, img_width)
    network = 'IndoorHall_5GHz'; Band = 'Wideband'; Link = 'Single'; scenario = 'LOS';
    freq = [-10e6 10e6] + 5.3e9; snapRate = 1; snapNum = img_height;
    BSPos = [10 10 0]; MSVelo = [0 0.001 0];
    [paraEx, paraSt, link, env, BS, MS] = cost2100(network, scenario, Link, Band, freq, snapRate, snapNum, BSPos, MSPos, MSVelo);
    delta_f = 7.8125e4;
    h_omni = create_IR_omni(link, freq, delta_f, Band);
    H_freq = fft(h_omni, [], 2);
    H_complex = H_freq(1:img_height, 1:img_width);
end

function MSPos = sample_user_position(dataset_name)
    switch dataset_name
        case 'user_center'
            x = 10 + (-2 + 4 * rand()); y = 5 + (-2 + 4 * rand());
        case 'user_edge'
            x = 10 + 8 * sign(rand() - 0.5) + randn(); y = 5 + 8 * sign(rand() - 0.5) + randn();
        case 'user_uniform'
            x = 2 + 16 * rand(); y = 1 + 8 * rand();
        case 'user_left_cluster'
            x = 4 + 1.5 * randn(); y = 5 + 1.5 * randn();
        case 'user_right_cluster'
            x = 16 + 1.5 * randn(); y = 5 + 1.5 * randn();
        case 'user_ring'
            r = 5 + randn(); theta = 2 * pi * rand(); x = 10 + r * cos(theta); y = 5 + r * sin(theta);
        otherwise
            error('Unknown dataset name.');
    end
    MSPos = [x y 0];
end
