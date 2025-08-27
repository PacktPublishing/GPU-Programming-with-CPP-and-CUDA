<h1 align="center">
GPU Programming with C++ and CUDA, First Edition</h1>
<p align="center">This is the code repository for <a href ="https://www.packtpub.com/en-us/product/gpu-programming-with-c-and-cuda-first-edition/9781805124542"> GPU Programming with C++ and CUDA, First Edition</a>, published by Packt.
</p>

<h2 align="center">
Uncover effective techniques for writing efficient GPU-parallel C++ applications
</h2>
<p align="center">
Paulo Motta</p>

<p align="center">
  <a href="https://packt.link/free-ebook/9781805124542"><img width="32px" alt="Free PDF" title="Free PDF" src="https://cdn-icons-png.flaticon.com/512/4726/4726010.png"/></a>
 &#8287;&#8287;&#8287;&#8287;&#8287;
  <a href="https://packt.link/gbp/9781805124542"><img width="32px" alt="Graphic Bundle" title="Graphic Bundle" src="https://cdn-icons-png.flaticon.com/512/2659/2659360.png"/></a>
  &#8287;&#8287;&#8287;&#8287;&#8287;
   <a href="https://www.amazon.com/GPU-Programming-CUDA-GPU-parallel-applications/dp/1805124544"><img width="32px" alt="Amazon" title="Get your copy" src="https://cdn-icons-png.flaticon.com/512/15466/15466027.png"/></a>
  &#8287;&#8287;&#8287;&#8287;&#8287;
</p>
<details open> 
  <summary><h2>About the book</summary>
<a href="https://www.packtpub.com/product/unity-cookbook-fifth-edition/9781805123026">
<img src="https://content.packt.com/B20897/cover_image_small.jpg" alt="Unity Cookbook, Fifth Edition" height="256px" align="right">
</a>

Written by Paulo Motta, a senior researcher with decades of experience, this comprehensive GPU programming book is an essential guide for leveraging the power of parallelism to accelerate your computations. The first section introduces the concept of parallelism and provides practical advice on how to think about and utilize it effectively. Starting with a basic GPU program, you then gain hands-on experience in managing the device. This foundational knowledge is then expanded by parallelizing the program to illustrate how GPUs enhance performance.

The second section explores GPU architecture and implementation strategies for parallel algorithms, and offers practical insights into optimizing resource usage for efficient execution.
In the final section, you will explore advanced topics such as utilizing CUDA streams. You will also learn how to package and distribute GPU-accelerated libraries for the Python ecosystem, extending the reach and impact of your work.

Combining expert insight with real-world problem solving, this book is a valuable resource for developers and researchers aiming to harness the full potential of GPU computing. The blend of theoretical foundations, practical programming techniques, and advanced optimization strategies it offers is sure to help you succeed in the fast-evolving field of GPU programming.</details>
<details open> 
  <summary><h2>Key Learnings</summary>
<ul>

<li>Manage GPU devices and accelerate your applications</li>

<li>Apply parallelism effectively using CUDA and C++</li>

<li>Choose between existing libraries and custom GPU solutions</li>

<li>Package GPU code into libraries for use with Python</li>

<li>Explore advanced topics such as CUDA streams</li>

<li>Implement optimization strategies for resource-efficient execution</li>

</ul>

  </details>

<details open> 
  <summary><h2>Chapters</summary>
     <img src="https://cliply.co/wp-content/uploads/2020/02/372002150_DOCUMENTS_400px.gif" alt="Unity Cookbook, Fifth Edition" height="556px" align="right">
<ol>

  <li>Introduction to Parallel Programming</li>

  <li>Setting Up Your Development Environment</li>

  <li>Hello CUDA</li>

  <li>Hello Again, but in Parallel</li>

  <li>A Closer Look into the World of GPUs</li>

  <li>Parallel Algorithms with CUDA</li>

  <li>Performance Strategies</li>

  <li>Overlaying Multiple Operations</li>

  <li>Exposing Your Code to Python</li>

  <li>Exploring Existing GPU Models</li>

</ol>

</details>


<details open> 
  <summary><h2>Requirements for this book</summary>
You should be comfortable writing computer programs in C++, and basic knowledge of operating systems will help to understand some of the more advanced concepts, given that we have to manage device communication.
<table>     
   <tr><td><b>Software / hardware covered in the book</td><td><b>Operating system requirements</td></tr>
   <tr><td>NVIDIA GPU or access to a Cloud-based VM with NVIDIA GPU</td><td>Ubuntu Linux 20 or later with NVIDIA Video Driver</td></tr>
   <tr><td>CUDA Toolkit 12</td><td></td></tr>
   <tr><td>Docker 27.0</td><td></td></tr>
   <tr><td>VS Code 1.92</td><td></td></tr>
   <tr><td>CMake 3.16</td><td></td></tr>
   <tr><td>g++ 9.4</td><td></td></tr>
   <tr><td>Python 3.8</td><td></td></tr>
   <tr><td>Nsight Compute 2023.3</td><td></td></tr>
</table>
  </details>

## Code conventions

We are using the following convetions:

1. camelCase for names of functions, kernels and variables
2. CamelCase with uppercase letter at beginning for structs
3. snake_case for names of files
4. When using two functions to perform the same computation on the CPU and GPU we use the same name with a Cpu/Kernel suffix like: computeSomethingCpu / computeSomethingKernel
5. When we need to allocate buffers with similar names, we are using h_ preffix for host side and d_for device side.
    * float* h_A;
    * float* d_A;
6. When comparing results we add a suffix to the name of the variable _CPU or _GPU.
    * float* h_C_GPU; // this is the C array or matrix calculated on the GPU and copied back to the host
    * float* h_C_CPU; // this is the C array or matrix calculated on the CPU

<details> 
  <summary><h2>Get to know author</h2></summary>

_Paulo Motta_ completed the PhD in Computer Science with an emphasis in parallel systems at PUC-Rio in 2011. Currently, Paulo Motta is a Senior Research Software Development Engineer at Microsoft and a postdoctoral researcher on quantum walks simulations with Hiperwalk at the National Scientific Computing Laboratory in Brazil. Paulo is a senior member of IEEE Computer Society with over 25 years' experience in software development and 9 years experience as a university professor.



</details>
<details> 
  <summary><h2>Other related books</h2></summary>
<ul>

  <li><a href="https://www.packtpub.com/en-us/product/c-high-performance-second-edition/9781839216541">C++ High Performance, Second Edition</a></li>

  <li><a href="https://www.packtpub.com/en-us/product/c-memory-management-first-edition/9781805129806">C++ Memory Management, First Edition</a></li>
 
</ul>

</details>
