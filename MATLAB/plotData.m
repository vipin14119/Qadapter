function plotData( X, y, levels )
%PLOTDATA plots the data set provided accorss value in y
%   Detailed explanation goes here
    figure;hold on;
    disp('------------ Plotting data -------------');
    title('Aptence Level dataset distribution');
    xlabel('Average Time (1-60)');
    ylabel('Average Accuracy (1-100)');
    CM = jet(length(levels));
    for l=levels
        class = find(y == l);
        plot(X(class, 1), X(class,2),'x','lineWidth', 2,'color', CM(l,:) ,'MarkerSize', 5, ...
            'DisplayName', sprintf('Level: %d', l));
    end
    legend('-DynamicLegend');
    hold off;

end
