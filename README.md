# SparkRenderer
Attempt to develop game engine using OpenGL 4.5

<img src="https://i.stack.imgur.com/gW6hR.jpg"/>

<b>Work done so far:</b>
<ul>
    <li>Scene serialization using compile time reflection from RTTR library.</li>
    <li>Reversed Z Depth (to increase depth buffer accuracy), ability to create infinite zFar projection.</li>
    <li>Encoded Normals using RG16F in G-Buffer.</li>
    <li>Resigned using one vertex buffer with interleaved vertex attributes (with stride and offsets) for many buffers
        which contain only one vertex attribute per buffer.</li>
    <li>Small G-Buffer (104 bits per pixel).</li>
    <ul>
        <li> Color R8G8B8A8 (A8 unused but needed to bind as an image to compute shader).</li>
        <li> Normals R16G16.</li>
        <li> Roughness and metalness R8G8.</li>
        <li> Depth texture 24b.</li>
    </ul>
    <li>PBR.</li>
    <li>IBL.</li>
    <li>Light Probes (work in progress).</li>
    <li>Parallax occlusion mapping.</li>
    <li>Tile-based light culling using compute shader.</li>
    <li>Tile-based deferred lighting using compute shader.</li>
    <li>Skybox.</li>
    <li>Bloom (in the Call of Duty: Advanced Warfare way).</li>
    <li>Camera movement Motion Blur.</li>
    <li>Tone Mapping (ACES filmic curve) with Eye Adaptation.</li>
    <li>FXAA.</li>
    <li>SSAO.</li>
    <li>Ability to load compressed textures in BCn compressed formats.
        Textures are compressed offline before running the engine using the Compressonator application.</li>
    <li>Strong usage of SSBOs to transfer and hold data.</li>
</ul>

<b>Thirdparty:</b>
<ul>
<li>glm</li>
<li>gli</li>
<li>GLFW</li>
<li>glad</li>
<li>assimp</li>
<li>json</li>
<li>object_threadsafe</li>
<li>spdlog</li>
<li>stb_image</li>
</ul>

<b>Compilation:</b>
The project is developed and compiled on Windows 10 platform with Visual Studio IDE.
