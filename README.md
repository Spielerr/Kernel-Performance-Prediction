# Kernel-Performance-Prediction
This project introduces an analytical model designed to predict the execution times of GPU kernels for unknown problem sizes without actual execution. The model considers various metrics involved in launching kernels, including problem size, block size, and grid size configurations, enabling the prediction of expected execution times.

# Abstract
With the deceleration of Moore's Law and the attenuated progress in hardware technology, the
imperative now lies in the development of efficient and parallel software to mitigate the scarcity in
hardware capabilities. In recent years, the demand for processing vast amounts of data across diverse
domains has attained paramount significance. A considerable focus has been directed towards
advancements in Machine Learning and Deep Learning, necessitating intensive data computations.
Simultaneously, the scientific domain has experienced a substantial surge in data collection
methodologies, precipitating a necessity for intricate data calculations. Consequently, there has been
an exponential surge in the utilization and evolution of Graphics Processing Units (GPUs).


While Central Processing Units (CPUs) were hitherto the benchmark for diverse computations, their
inherent limitations in handling massive parallel applications have paved the way for General
Purpose GPUs (GPGPUs). GPU applications are now pervasive and find application across various
domains, yielding unprecedented performance enhancements and significant cost reductions due to
GPUs' superior power efficiency per computation compared to CPUs.


Nevertheless, the development of GPU code is a non-trivial undertaking. Software developers,
accustomed to learning algorithms and general software development in a sequential manner, must
invest time and effort in transitioning to parallel thinking. Consequently, planning and developing
massively parallel GPU code entail substantial cost implications. Furthermore, GPUs are applicable
only to specific problem types that exhibit notable characteristics conducive to parallelization.


Identifying a problem suitable for GPU utilization raises questions regarding the potential
performance improvement compared to traditional CPU execution without parallelism. Thus,
predicting the performance of GPU code (analogous to kernels on GPUs) becomes crucial in
quantifying potential enhancements resulting from the parallelization of software for GPUs. This
report introduces an analytical model designed to predict the execution times of GPU kernels for
unknown problem sizes without actual execution. The model considers various metrics involved in
launching kernels, including problem size, block size, and grid size configurations, enabling the
prediction of expected execution times. It is also shown that the prediction can be extended to
different GPUs, considering their own hardware characteristics. Consequently, the model instills a
degree of confidence without necessitating an in-depth understanding of kernel intricacies, their
functionality, or how they scale with increasing problem sizes. The model automates this process,
providing a streamlined approach to performance prediction. For the simplicity of the model, it offers
adequate prediction accuracy and provides good scope for future expansion to consider more niche
parameters affecting all kinds of kernels and subsequently their performance.
