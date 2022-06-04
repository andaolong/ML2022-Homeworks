% 绘制图像的函数


function plotData(x, y)
    figure; % 开启一个新图表
    plot(x, y, 'rx', 'MarkerSize', 5); % 这些选项的含义写下面了
    xlabel("Population of city in 10000s"); % 指定图表坐标轴的标签
    ylabel('Profit in $10000s');
end


% plot(x, y, 'rx', 'MarkerSize', 5); 
% 这句话的含义：
% plot(x, y)默认是绘制折线图
% 加上'rx'选项，从折线图变成了红色的散点图，
% r表示red,x表示十字号x,表示散点图的点是红色的十字符号x
% 加上"MarkerSize",10, 散点图的点变大了


%PLOTDATA Plots the data points x and y into a new figure
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.
% ====================== YOUR CODE HERE ======================
% Instructions: Plot the training data into a figure using the
%               "figure" and "plot" commands. Set the axes labels using
%               the "xlabel" and "ylabel" commands. Assume the
%               population and revenue data have been passed in
%               as the x and y arguments of this function.
%
% Hint: You can use the 'rx' option with plot to have the markers
%       appear as red crosses. Furthermore, you can make the
%       markers larger by using plot(..., 'rx', 'MarkerSize', 10);
% ============================================================
