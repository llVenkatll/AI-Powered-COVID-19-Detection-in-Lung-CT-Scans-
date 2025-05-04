%% Advanced COVID-19 Detection in Lung CT Scans
% This script implements a sophisticated image processing pipeline for COVID-19 detection
% in lung CT scans using adaptive techniques, multi-scale analysis, and machine learning integration

% Clear environment
clc;
clear all;
close all;

%% Configuration Parameters
% Adjustable parameters for processing optimization
config = struct();
config.denoise = struct('method', 'gaussian', 'sigma', 1.8, 'size', [5 5]);
config.clahe = struct('clip_limit', 0.02, 'tiles', [8 8]);
config.sharpen = struct('radius', 2.5, 'amount', 1.8, 'threshold', 0.4);
config.edge = struct('method', 'canny', 'threshold', [0.04 0.10], 'sigma', 1.5);
config.morph = struct('close_radius', 3, 'open_radius', 2, 'fill_holes', true);
config.segment = struct('num_clusters', 4, 'distance', 'cityblock', 'min_area', 100);
config.overlap_threshold = 0.3;  % For region merging
config.feature_extraction = true; % Enable feature extraction for ML

%% Load and Preprocess Image
% Read the input image with error handling
try
    % Define image path with flexibility
    base_path = pwd;
    
    % Option 1: Use file dialog to select image interactively
    [file, path] = uigetfile({'*.jpg;*.png;*.tif;*.dcm;*.bmp', 'Image Files'}, 'Select CT Scan');
    if file == 0
        fprintf('No file selected. Using a default filename.\n');
        % Option 2: Use a default filename if no file selected
        image_path = fullfile(base_path, 'lung_ct.jpg');
    else
        image_path = fullfile(path, file);
    end
    
    fprintf('Loading image from: %s\n', image_path);
    
    % Read the image
    I = imread("C:\Users\venka\Downloads\lungcovid.jpg");
    
    % Display original image
    figure('Name', 'Original CT Scan', 'NumberTitle', 'off');
    imshow(I);
    title('Original CT Scan');
    
    % Check and convert to grayscale if RGB
    if size(I, 3) == 3
        I_gray = rgb2gray(I);
        fprintf('Converted RGB image to grayscale\n');
    else
        I_gray = I;
        fprintf('Image is already grayscale\n');
    end
    
    % Check image type and convert to double for processing
    I_gray = im2double(I_gray);
    
catch ME
    fprintf('Error loading image: %s\n', ME.message);
    fprintf('To run this code, please provide a CT scan image file.\n');
    return;
end

%% Image Analysis and Enhancement
% Create a processing history for comparison
proc_history = struct();
proc_history.original = I_gray;

% Global image analysis
[rows, cols] = size(I_gray);
fprintf('Image dimensions: %d x %d\n', rows, cols);

% Calculate image statistics
global_mean = mean(I_gray(:));
global_std = std(I_gray(:));
global_min = min(I_gray(:));
global_max = max(I_gray(:));
global_entropy = entropy(I_gray);

fprintf('Global statistics:\n');
fprintf('  Mean: %.4f\n  Std: %.4f\n  Min: %.4f\n  Max: %.4f\n  Entropy: %.4f\n', ...
    global_mean, global_std, global_min, global_max, global_entropy);

% Apply lung mask (simplified version - would be more advanced in production)
% This would be replaced with proper lung segmentation algorithms
lung_mask = I_gray > (global_mean - 0.5*global_std) & I_gray < (global_mean + 2*global_std);
lung_mask = imopen(lung_mask, strel('disk', 15));
lung_mask = imclose(lung_mask, strel('disk', 25));
lung_mask = imfill(lung_mask, 'holes');

% Apply adaptive histogram equalization to enhance lung structures
I_clahe = adapthisteq(I_gray, 'ClipLimit', config.clahe.clip_limit, ...
    'NumTiles', config.clahe.tiles);
proc_history.clahe = I_clahe;

% Apply bilateral filtering to preserve edges while removing noise
I_bilateral = bilateralFilter(I_clahe, 5, 0.3, 1.5);
proc_history.bilateral = I_bilateral;

%% Multi-scale Analysis
% Perform analysis at multiple scales to capture different-sized features
scales = [1, 2, 4];
multi_scale_features = cell(length(scales), 1);

for scale_idx = 1:length(scales)
    scale = scales(scale_idx);
    
    % Downsample image for this scale
    if scale > 1
        I_scaled = imresize(I_bilateral, 1/scale);
    else
        I_scaled = I_bilateral;
    end
    
    % Apply Gabor filter bank at this scale for texture analysis
    wavelength = 4 * scale;
    orientation_count = 6;
    gabor_features = extractGaborFeatures(I_scaled, wavelength, orientation_count);
    
    % Store features for this scale
    multi_scale_features{scale_idx} = gabor_features;
    
    % Enhance edges at this scale
    I_edge_enhanced = enhanceEdges(I_scaled, config.edge);
    
    % Restore to original size if needed
    if scale > 1
        I_edge_enhanced = imresize(I_edge_enhanced, size(I_bilateral));
    end
    
    % Store in history with scale info
    field_name = sprintf('edge_scale_%d', scale);
    proc_history.(field_name) = I_edge_enhanced;
end

%% Block-Based Processing with Adaptive Parameters
% Determine the size of each block with overlap
block_size = [floor(rows/4), floor(cols/4)];
overlap = floor(block_size/3);  % 33% overlap between blocks

% Calculate number of blocks in each dimension
num_blocks_rows = ceil((rows - overlap(1)) / (block_size(1) - overlap(1)));
num_blocks_cols = ceil((cols - overlap(2)) / (block_size(2) - overlap(2)));

% Initialize matrices to store block-specific results
block_results = zeros(rows, cols);
block_features = cell(num_blocks_rows, num_blocks_cols);
block_anomaly_scores = zeros(num_blocks_rows, num_blocks_cols);

fprintf('Processing %d x %d blocks with %d%% overlap...\n', ...
    num_blocks_rows, num_blocks_cols, round(100*overlap(1)/block_size(1)));

% Process each block with adaptive parameters
for i = 1:num_blocks_rows
    for j = 1:num_blocks_cols
        % Calculate block coordinates with overlap
        row_start = max(1, (i-1) * (block_size(1) - overlap(1)) + 1);
        row_end = min(rows, row_start + block_size(1) - 1);
        col_start = max(1, (j-1) * (block_size(2) - overlap(2)) + 1);
        col_end = min(cols, col_start + block_size(2) - 1);
        
        % Extract current block
        I_block = I_bilateral(row_start:row_end, col_start:col_end);
        block_mask = lung_mask(row_start:row_end, col_start:col_end);
        
        % Skip processing if the block has little lung tissue
        lung_percentage = sum(block_mask(:)) / numel(block_mask);
        if lung_percentage < 0.2
            fprintf('Block (%d,%d) skipped - insufficient lung tissue (%.1f%%)\n', ...
                i, j, 100*lung_percentage);
            continue;
        end
        
        % Analyze block statistics for adaptive parameter adjustment
        block_mean = mean(I_block(block_mask));
        block_std = std(I_block(block_mask));
        block_entropy = entropy(I_block);
        
        % Adjust parameters based on block characteristics
        local_config = adaptConfigToBlock(config, block_mean, block_std, block_entropy);
        
        % Apply processing pipeline to the block
        % 1. Enhanced denoising with edge preservation
        if strcmp(local_config.denoise.method, 'gaussian')
            I_denoised = imgaussfilt(I_block, local_config.denoise.sigma);
        else
            I_denoised = wiener2(I_block, local_config.denoise.size);
        end
        
        % 2. Local contrast enhancement
        I_contrast = adapthisteq(I_denoised, 'ClipLimit', local_config.clahe.clip_limit);
        
        % 3. Adaptive sharpening based on block characteristics
        I_sharpened = locallyAdaptiveSharpening(I_contrast, block_mask, local_config.sharpen);
        
        % 4. Texture feature extraction
        if config.feature_extraction
            block_texture_features = extractTextureFeatures(I_sharpened, block_mask);
            % Store features for potential machine learning
            block_features{i,j} = block_texture_features;
        end
        
        % 5. Multi-scale edge detection
        I_edges = detectEdgesMultiScale(I_sharpened, local_config.edge);
        
        % 6. Morphological operations with adaptive structuring elements
        se_size = max(2, round(min(block_size)/40));
        se_close = strel('disk', se_size);
        I_closed = imclose(I_edges, se_close);
        
        % 7. Advanced region segmentation using clustering
        [I_segmented, regionProps] = segmentRegions(I_sharpened, I_closed, ...
            local_config.segment, block_mask);
        
        % 8. Calculate anomaly score based on region properties
        anomaly_score = calculateAnomalyScore(regionProps, block_texture_features);
        block_anomaly_scores(i,j) = anomaly_score;
        
        % Store processed result
        block_results(row_start:row_end, col_start:col_end) = ...
            block_results(row_start:row_end, col_start:col_end) + I_segmented;
            
        % Display block processing results (reduced for batch processing)
        if i == 2 && j == 2  % Display middle block as example
            figure('Name', sprintf('Block Processing (%d,%d)', i, j), 'NumberTitle', 'off');
            subplot(2,3,1), imshow(I_block), title('Original Block');
            subplot(2,3,2), imshow(I_sharpened), title('Enhanced');
            subplot(2,3,3), imshow(I_edges), title('Edge Detection');
            subplot(2,3,4), imshow(I_closed), title('Morphological');
            subplot(2,3,5), imshow(I_segmented), title('Segmented');
            subplot(2,3,6), imshow(block_mask), title('Lung Mask');
        end
    end
end

% Normalize the results for overlapping regions
block_results = block_results / max(block_results(:));

%% Visualization and Analysis of Results
% Create heatmap of anomaly scores
figure('Name', 'COVID-19 Detection Results', 'NumberTitle', 'off');

% Original with segmentation overlay
subplot(2,2,1);
imshow(I_gray);
hold on;
h = imagesc(block_results);
set(h, 'AlphaData', 0.5);
colormap(gca, jet);
title('Original with Segmentation Overlay');

% Anomaly score heatmap
subplot(2,2,2);
anomaly_heatmap = imresize(block_anomaly_scores, size(I_gray), 'bicubic');
imagesc(anomaly_heatmap);
colormap(gca, hot);
colorbar;
title('Anomaly Score Heatmap (Higher = Potential COVID-19)');

% 3D view of the anomaly scores
subplot(2,2,3:4);
[X, Y] = meshgrid(1:size(anomaly_heatmap,2), 1:size(anomaly_heatmap,1));
surf(X, Y, anomaly_heatmap, 'EdgeColor', 'none');
colormap(jet);
colorbar;
title('3D Visualization of Anomaly Distribution');
xlabel('X Coordinate'); 
ylabel('Y Coordinate');
zlabel('Anomaly Score');
view(30, 45);

%% Quantitative Analysis and Decision Support
% Apply thresholding to determine potential COVID-19 regions
threshold = 0.65;  % This would be determined through validation
potential_covid_regions = anomaly_heatmap > threshold;

% Calculate statistics for decision support
total_anomaly_area = sum(potential_covid_regions(:));
percent_affected = 100 * total_anomaly_area / sum(lung_mask(:));
max_anomaly_score = max(anomaly_heatmap(:));

% Display analysis results
fprintf('\nDetection Results:\n');
fprintf('  Total potential COVID-19 affected area: %d pixels\n', total_anomaly_area);
fprintf('  Percentage of lung affected: %.2f%%\n', percent_affected);
fprintf('  Maximum anomaly score: %.2f\n', max_anomaly_score);

% Decision suggestion (simplified - would be based on validated thresholds)
if percent_affected > 5 && max_anomaly_score > 0.8
    fprintf('\nSuggestion: High probability of COVID-19 infection.\n');
elseif percent_affected > 2 || max_anomaly_score > 0.7
    fprintf('\nSuggestion: Moderate probability of COVID-19 infection. Further examination recommended.\n');
else
    fprintf('\nSuggestion: Low probability of COVID-19 infection based on CT scan.\n');
end

%% Supporting Functions

function filtered = bilateralFilter(img, sigma_s, sigma_r, window_size)
    % Bilateral filter implementation with proper error checking
    [rows, cols] = size(img);
    filtered = img; % Initialize with original image to handle border cases
    
    % Check if image has enough pixels to process
    if rows <= 2*window_size+1 || cols <= 2*window_size+1
        fprintf('Warning: Image too small for bilateral filter with window size %d. Skipping filter.\n', window_size);
        return;
    end
    
    % Create spatial kernel
    [x, y] = meshgrid(-window_size:window_size, -window_size:window_size);
    spatial_kernel = exp(-(x.^2 + y.^2)/(2*sigma_s^2));
    
    % Apply bilateral filter with proper bounds checking
    for i = 1+window_size:rows-window_size
        for j = 1+window_size:cols-window_size
            % Extract window
            window = img(i-window_size:i+window_size, j-window_size:j+window_size);
            
            % Calculate range kernel
            center_value = img(i, j);
            range_kernel = exp(-(window - center_value).^2/(2*sigma_r^2));
            
            % Calculate combined filter weight
            combined_kernel = spatial_kernel .* range_kernel;
            
            % Check for numerical issues
            kernel_sum = sum(combined_kernel(:));
            if kernel_sum > 0
                combined_kernel = combined_kernel / kernel_sum;
                % Apply filter
                filtered(i, j) = sum(window(:) .* combined_kernel(:));
            end
        end
    end
end

function features = extractGaborFeatures(img, wavelength, orientationCount)
    % Extract Gabor features at multiple orientations
    features = zeros(size(img,1), size(img,2), orientationCount);
    
    for i = 1:orientationCount
        % Calculate orientation in radians
        theta = (i-1) * pi / orientationCount;
        
        % Create Gabor filter
        gabor_size = ceil(wavelength * 2.5);
        sigma_x = gabor_size / 2;
        sigma_y = sigma_x * 0.7;
        
        [x, y] = meshgrid(-gabor_size:gabor_size, -gabor_size:gabor_size);
        
        % Rotate coordinates
        x_theta = x * cos(theta) + y * sin(theta);
        y_theta = -x * sin(theta) + y * cos(theta);
        
        % Create filter
        gb = exp(-0.5 * (x_theta.^2/sigma_x^2 + y_theta.^2/sigma_y^2)) .* ...
             cos(2 * pi * x_theta / wavelength);
        
        % Normalize filter
        gb = gb - mean(gb(:));
        gb = gb / sqrt(sum(gb(:).^2));
        
        % Apply filter
        features(:,:,i) = conv2(img, gb, 'same');
    end
end

function enhanced = enhanceEdges(img, edge_config)
    % Enhance edges based on configuration
    
    % Apply edge detection
    if strcmp(edge_config.method, 'canny')
        edges = edge(img, 'canny', edge_config.threshold, edge_config.sigma);
    else
        edges = edge(img, 'sobel');
    end
    
    % Enhance original image based on edges
    enhanced = img;
    edge_mask = imdilate(edges, strel('disk', 1));
    enhanced(edge_mask) = enhanced(edge_mask) * 1.2;
    
    % Normalize
    enhanced = (enhanced - min(enhanced(:))) / (max(enhanced(:)) - min(enhanced(:)));
end

function local_config = adaptConfigToBlock(config, block_mean, block_std, block_entropy)
    % Adapt configuration parameters based on block statistics
    local_config = config;
    
    % Adjust denoising based on noise level (measured by std)
    if block_std > 0.15
        local_config.denoise.sigma = config.denoise.sigma * 1.2;
    else
        local_config.denoise.sigma = config.denoise.sigma * 0.8;
    end
    
    % Adjust CLAHE based on contrast
    if block_std < 0.1
        local_config.clahe.clip_limit = config.clahe.clip_limit * 1.5;
    end
    
    % Adjust edge detection based on entropy
    if block_entropy > 6.5
        local_config.edge.threshold = config.edge.threshold * 1.2;
    else
        local_config.edge.threshold = config.edge.threshold * 0.8;
    end
end

function enhanced = locallyAdaptiveSharpening(img, mask, sharpen_config)
    % Apply sharpening with locally adaptive parameters
    
    % Create a sharpened version
    enhanced = imsharpen(img, 'Radius', sharpen_config.radius, ...
        'Amount', sharpen_config.amount, 'Threshold', sharpen_config.threshold);
    
    % Apply mask if provided
    if nargin > 1 && ~isempty(mask)
        enhanced = img .* (~mask) + enhanced .* mask;
    end
end

function features = extractTextureFeatures(img, mask)
    % Extract texture features for potential machine learning
    
    % Apply mask if provided
    if nargin > 1 && ~isempty(mask)
        img_masked = img .* mask;
    else
        img_masked = img;
    end
    
    % Calculate GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(im2uint8(img_masked), 'Offset', [0 1; -1 1; -1 0; -1 -1], ...
        'NumLevels', 32, 'Symmetric', true);
    
    % Extract GLCM properties
    stats = graycoprops(glcm, {'contrast', 'correlation', 'energy', 'homogeneity'});
    
    % LBP (Local Binary Pattern) - simplified version
    lbp_features = extractLBP(img);
    
    % Combine features
    features = struct();
    features.mean = mean(img_masked(mask));
    features.std = std(img_masked(mask));
    features.entropy = entropy(img_masked);
    features.contrast = mean([stats.Contrast]);
    features.correlation = mean([stats.Correlation]);
    features.energy = mean([stats.Energy]);
    features.homogeneity = mean([stats.Homogeneity]);
    features.lbp = lbp_features;
end

function lbp = extractLBP(img)
    % Simple LBP implementation
    [rows, cols] = size(img);
    lbp = zeros(rows-2, cols-2);
    
    for i = 2:rows-1
        for j = 2:cols-1
            center = img(i, j);
            code = 0;
            code = code + (img(i-1, j-1) >= center) * 1;
            code = code + (img(i-1, j  ) >= center) * 2;
            code = code + (img(i-1, j+1) >= center) * 4;
            code = code + (img(i  , j+1) >= center) * 8;
            code = code + (img(i+1, j+1) >= center) * 16;
            code = code + (img(i+1, j  ) >= center) * 32;
            code = code + (img(i+1, j-1) >= center) * 64;
            code = code + (img(i  , j-1) >= center) * 128;
            lbp(i-1, j-1) = code;
        end
    end
    
    % Compute histogram of LBP codes
    lbp_hist = histcounts(lbp(:), 0:256) / numel(lbp);
    
    % Return histogram as feature
    lbp = lbp_hist;
end

function edges = detectEdgesMultiScale(img, edge_config)
    % Multi-scale edge detection
    
    % Standard edge detection
    edges1 = edge(img, edge_config.method, edge_config.threshold);
    
    % Coarser scale edge detection
    img_coarse = imgaussfilt(img, 2);
    edges2 = edge(img_coarse, edge_config.method, edge_config.threshold * 0.8);
    
    % Combine edges from multiple scales
    edges = edges1 | edges2;
    
    % Clean up with morphological operations
    edges = bwareaopen(edges, 5);  % Remove tiny segments
end

function [segmented, regionProps] = segmentRegions(img, edges, segment_config, mask)
    % Segment regions based on edge information and clustering
    
    % Create markers for watershed using edges
    edge_mask = ~edges;
    if nargin > 3 && ~isempty(mask)
        edge_mask = edge_mask & mask;
    end
    
    % Distance transform for watershed markers
    D = bwdist(edges);
    D = imimposemin(D, imregionalmax(D));
    
    % Apply watershed
    L = watershed(D);
    
    % Apply clustering within each region
    segmented = zeros(size(img));
    
    % Get region properties
    labeled_regions = label2rgb(L);
    regionProps = regionprops(L, img, 'Area', 'MeanIntensity', 'MaxIntensity', 'MinIntensity', ...
        'Perimeter', 'Eccentricity');
    
    % Filter regions by size
    areas = [regionProps.Area];
    valid_regions = areas > segment_config.min_area;
    
    % Compute a segmentation score for each valid region
    for i = 1:length(regionProps)
        if valid_regions(i)
            region_label = i;
            region_mask = L == region_label;
            region_intensity = regionProps(i).MeanIntensity;
            
            % Assign a score based on intensity and other properties
            % Higher score for unusual (potentially pathological) regions
            intensity_score = abs(region_intensity - mean(img(:))) / std(img(:));
            shape_score = regionProps(i).Perimeter^2 / (4 * pi * regionProps(i).Area);
            
            region_score = intensity_score * shape_score;
            
            % Apply the score to the segmentation
            segmented(region_mask) = min(1, region_score);
        end
    end
end

function score = calculateAnomalyScore(regionProps, textureFeatures)
    % Calculate an anomaly score based on region and texture properties
    
    if isempty(regionProps)
        score = 0;
        return;
    end
    
    % Extract relevant metrics
    areas = [regionProps.Area];
    intensities = [regionProps.MeanIntensity];
    
    % Compute basic anomaly indicators
    avg_intensity = mean(intensities);
    intensity_deviation = std(intensities);
    
    % COVID-19 typically shows as ground-glass opacity and consolidation
    % This would be areas of increased intensity with specific texture patterns
    large_bright_regions = sum(areas(intensities > avg_intensity + 0.5*intensity_deviation));
    
    % Combine with texture features if available
    if nargin > 1 && ~isempty(textureFeatures)
        texture_score = textureFeatures.correlation * textureFeatures.entropy / ...
            (textureFeatures.energy + 0.001);
        
        % Combine scores (weights would be determined through validation)
        score = 0.6 * (large_bright_regions / sum(areas)) + 0.4 * texture_score;
    else
        score = large_bright_regions / sum(areas);
    end
    
    % Normalize score to 0-1 range with a non-linear mapping
    score = min(1, score * 2);
end