Real-time-RGB-D-data-processing-on-GPU-architecture
===================================================

Efficient system for real-time RGB-D camera data processing on GPU architecture.

Its goal is to improve the depth data accuracy while processing the RGB-D data stream in real time, thus being very attractive for depth-based interactive applications such as gesture recognition for human-computer interaction, 3D scene modeling, etc. The proposed system performs a pixel-wise fusion of depth and color data based on adaptive filtering and background modeling techniques, which guarantees an efficient parallelization on any GPU architecture. Results prove that the average throughput, of around 200 fps, is well below those generally required by RGB-D cameras for real-time operation, despite the significant improvement of depth data accuracy.

The main function is in the "mixture.cpp" file


