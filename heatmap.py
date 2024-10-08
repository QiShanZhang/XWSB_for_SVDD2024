import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming we have some data for the heatmap
# This will be a 4x25 array of random numbers to simulate the data from the image
# You will replace this with your actual data
data = [[8.0113e-01,
4.8300e-01,
5.1007e-01,
4.2199e-01,
3.1268e-01,
1.9131e-01,
2.2563e-01,
1.2401e-01,
6.8804e-02,
3.8077e-02,
1.6757e-02,
8.5131e-04,
2.0785e-04,
9.6044e-06,
2.1213e-06,
5.1678e-08,
8.2690e-09,
2.3459e-08,
9.9277e-11,
1.0540e-13,
8.7618e-15,
1.4590e-22,
3.6035e-37,
6.8644e-35,
1.0000e+00
]
,
[
            7.6867e-01,
            5.0651e-01,
            4.8871e-01,
            4.4454e-01,
            2.2953e-01,
            1.6956e-01,
            1.7595e-01,
            1.5429e-01,
            3.4334e-01,
            9.7052e-01,
            9.9928e-01,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00
],
[
            8.0637e-01,
            5.6677e-01,
            5.3085e-01,
            4.0599e-01,
            2.9860e-01,
            2.4250e-01,
            2.0977e-01,
            2.5858e-01,
            6.7826e-01,
            9.9939e-01,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00
],
[
            7.6701e-01,
            4.7828e-01,
            4.3471e-01,
            4.2683e-01,
            2.7180e-01,
            2.4043e-01,
            2.3278e-01,
            2.8874e-01,
            6.5385e-01,
            9.9922e-01,
            9.9999e-01,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00
],[
            8.1875e-01,
            6.0886e-01,
            6.2466e-01,
            5.7925e-01,
            3.1178e-01,
            2.6506e-01,
            2.1797e-01,
            1.3976e-01,
            4.2443e-01,
            9.5427e-01,
            9.9961e-01,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00,
            1.0000e+00
],
        [0,2.3807e-01,
1.4623e-01,
1.4794e-01,
1.3121e-01,
9.6841e-02,
6.1204e-02,
8.7199e-02,
6.7863e-02,
6.2210e-02,
2.3732e-02,
3.2992e-02,
1.8970e-02,
1.5643e-02,
2.0920e-02,
2.5110e-02,
2.4200e-02,
7.1514e-03,
3.0650e-03,
1.5545e-03,
6.0369e-04,
6.6937e-06,
2.0090e-09,
9.9093e-08,
9.9788e-01],
        [0,2.2614e-01,
1.8072e-01,
1.9742e-01,
1.6174e-01,
1.3209e-01,
7.4589e-02,
1.4438e-01,
1.2325e-01,
7.8397e-02,
3.4990e-02,
5.5102e-02,
3.6046e-02,
5.0022e-02,
3.9390e-02,
1.0865e-01,
1.7208e-01,
3.0348e-01,
3.2312e-01,
7.4236e-01,
9.9840e-01,
9.9999e-01,
9.9999e-01,
9.9999e-01,
1.0000e+00],
        [0,2.6127e-01,
1.6724e-01,
1.6816e-01,
1.6774e-01,
1.3274e-01,
6.7229e-02,
1.2627e-01,
9.9572e-02,
5.9938e-02,
2.0778e-02,
3.8448e-02,
2.4482e-02,
2.5317e-02,
3.2725e-02,
8.4986e-02,
1.2581e-01,
1.6107e-01,
1.8156e-01,
6.6511e-01,
9.9746e-01,
9.9999e-01,
9.9999e-01,
1.0000e+00,
1.0000e+00],
        [0,2.6608e-01,
1.8115e-01,
2.0483e-01,
1.8974e-01,
1.4482e-01,
7.3923e-02,
1.5699e-01,
1.3667e-01,
8.3335e-02,
3.1863e-02,
5.5479e-02,
3.3976e-02,
4.3793e-02,
4.8320e-02,
1.3758e-01,
2.1506e-01,
2.8364e-01,
2.4343e-01,
7.9656e-01,
9.9859e-01,
1.0000e+00,
9.9999e-01,
9.9999e-01,
1.0000e+00],[0,2.3299e-01,
1.2890e-01,
1.6732e-01,
1.6891e-01,
1.2963e-01,
7.6103e-02,
1.3133e-01,
1.2796e-01,
9.6003e-02,
3.1701e-02,
4.8971e-02,
2.9127e-02,
2.8267e-02,
3.2113e-02,
7.3254e-02,
1.0788e-01,
1.7155e-01,
1.7705e-01,
6.5552e-01,
9.9433e-01,
9.9997e-01,
9.9995e-01,
9.9992e-01,
1.0000e+00]

        ]

# Assuming the row labels (challenges) and column labels (layers) are as follows:
# row_labels = [ 'sigmoid','sigmoid',
#                'softmax','softmax',
#                'sigmoid', 'sigmoid',
#                'softmax', 'softmax',
#                'sigmoid', 'sigmoid',
#                'softmax', 'softmax',]
column_labels = list(range(25)) # 0 to 24 layers

# Create the heatmap
plt.figure(figsize=(10, 4))
#yticklabels=row_labels
ax = sns.heatmap(data, xticklabels=column_labels, cmap='coolwarm',linecolor='black', linewidths=0.5)

# Add a colorbar with a label
cbar = ax.collections[0].colorbar
cbar.set_label('Value')

# Set the labels for axes
plt.xlabel('Layer')
# plt.ylabel('model')
plt.tight_layout()
# Show the plot
plt.savefig('heatmap.pdf', format='pdf')
plt.show()

