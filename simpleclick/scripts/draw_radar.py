import matplotlib.pyplot as plt
import numpy as np



def example_data():
    data = [
                # ('NoC@80%', [
                #     [1.36, 1.30, 2.82, 1.75, 2.62, 5.52, 2.15, 13.50, 1.77],
                #     [0 for _ in range(9)],
                #     [1.56, 1.56, 3.19, 2.41, 3.41, 6.10, 3.35, 13.31, 2.48],
                #     [0 for _ in range(9)],
                #     [0 for _ in range(9)]]),
                ('NoC@85%', [
                    [1.36, 1.40, 3.79, 1.94, 3.47, 7.18, 2.80, 16.38, 2.05],
                    [0 for _ in range(9)],
                    [1.70, 1.83, 4.41, 2.71, 4.46, 7.76, 4.05, 16.71, 2.90],
                    [0 for _ in range(9)],
                    [0 for _ in range(9)]]),
                ('NoC@90%', [
                    [1.50, 2.08, 5.11, 2.25, 5.54, 10.82, 5.55, 19.17, 2.79],
                    [0 for _ in range(9)],
                    [1.92, 2.79, 6.08, 3.17, 6.79, 11.24, 6.20, 19.09, 3.80],
                    [0 for _ in range(9)],
                    [0 for _ in range(9)]])
    ]
    return data


if __name__ == '__main__':
    N = 9
    models = ('SOTA', 'ViT-B-224', 'Vit-B-448', 'ViT-L-224', 'ViT-L-448')
    num_models = len(models)

    colors = ('#1aaf6c', '#429bf4', '#d42cea', '#feea11', '#00ffff')

    datasets = ['GrabCut', 'Berkeley', 'DAVIS', 'Pascal', 'SBD', 'BraTS', 'ssTEM', 'OAI-ZIB', 'COCO_MVal']
    num_datasets = len(datasets)

    angles = np.linspace(0, 2 * np.pi, num_datasets, endpoint=False).tolist()
    angles += angles[:1]

    fig, axs = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, subplot_kw=dict(polar=True))
    fig.subplots_adjust(wspace=0.5, hspace=0.05, top=0.85, bottom=0.05)

    def add_to_radar(label, values, color, alpha=0.25):
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=1, label=label)
        ax.fill(angles, values, color=color, alpha=alpha)

    metric_data = example_data()
    for ax, (metric, data) in zip(axs.flat, metric_data):

        # add each model to the chart
        for i, model in enumerate(models):
            values = data[i]
            add_to_radar(models[i], values, color=colors[i])

        ax.set_title(metric, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), datasets)

        # Go through labels and adjust alignment based on where
        # it is in the circle.
        for label, angle in zip(ax.get_xticklabels(), angles):
            if angle in (0, np.pi):
                label.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')

        ax.set_ylim(0, 20)

        # Add some custom styling.
        # Change the color of the tick labels.
        ax.tick_params(colors='#222222')
        # Make the y-axis (0-100) labels smaller.
        ax.tick_params(axis='y', labelsize=8)
        # Change the color of the circular gridlines.
        ax.grid(color='#AAAAAA')
        # Change the color of the outermost gridline (the spine).
        ax.spines['polar'].set_color('#222222')
        # Change the background color inside the circle itself.
        ax.set_facecolor('#FAFAFA')

    axs[0].legend(loc=(1.1, 0.95))

    # axs[0].legend(loc=(1.2, 0.95), labelspacing=0.1, fontsize='small')
    # fig.text(0.5, 0.965, 'Comparison Results',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='large')

    plt.show()