function [h, display_array] = displayData(X, example_width)
	%DISPLAYDATA Display 2D data in a nice grid
	%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
	%   stored in X in a nice grid. It returns the figure handle h and the 
	%   displayed array if requested.

	% Set example_width automatically if not passed in
	% 如果实参没有传进来example_width, 那么通过X的size计算出example_width
	if ~exist('example_width', 'var') || isempty(example_width) 
		example_width = round(sqrt(size(X, 2)));
	end

	% Gray Image
	colormap(gray);

	% Compute rows, cols
	[m n] = size(X);
	example_height = (n / example_width);

	% Compute number of items to display
	% 展示的时候将要展示的图像排布成一个正方形，所以行需要m取平方根
	display_rows = floor(sqrt(m));
	display_cols = ceil(m / display_rows);

	% Between images padding; 图片之间的间隔
	pad = 1;

	% Setup blank display
	% 构造展示矩阵，
	% 行数是 1+行图片数*(图片高度像素数+1)，
	% 列数是，1+列图片数*(图片宽度像素数+1)
	display_array = - ones(pad + display_rows * (example_height + pad), ...
						pad + display_cols * (example_width + pad));


	% Copy each example into a patch on the display array
	% 将原数据中每一张图片中的数据，排布到展示矩阵对应的patch补丁上
	curr_ex = 1;
	for j = 1:display_rows
		for i = 1:display_cols
			if curr_ex > m, 
				break; 
			end
			% Copy the patch
			
			% Get the max value of the patch
			max_val = max(abs(X(curr_ex, :)));
			display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
						pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
							reshape(X(curr_ex, :), example_height, example_width) / max_val;
			curr_ex = curr_ex + 1;
		end
		if curr_ex > m, 
			break; 
		end
	end

	% Display Image
	h = imagesc(display_array, [-1 1]);

	% Do not show axis
	axis image off

	drawnow;

end
