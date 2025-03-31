# Cross-Reality Representation Matrix

## Video Conference (VC) Participant's View

| Platform Type | How VR Users Appear | Capabilities | Limitations | Common Implementation |
|--------------|---------------------|--------------|-------------|---------------------|
| Zoom/Teams/Meet | 2D video feed of VR avatar or Mixed Reality view | - Facial expressions mapped to avatar\n- Hand gestures\n- Basic body movements | - Limited depth perception\n- No spatial interaction\n- Reduced presence | - Window into VR space\n- Picture-in-Picture of avatar\n- Mixed reality composite |
| Web Browser | 2D render of VR space and avatars | - Multiple viewpoint options\n- Shared workspace view\n- Interactive 3D elements | - No direct manipulation\n- Limited immersion\n- Passive viewing | - WebGL render\n- Interactive 3D viewport\n- Screen sharing |
| Mobile Device | Simplified avatar visualization | - Basic gesture following\n- Attention indicators\n- Status updates | - Small screen constraints\n- Limited interaction\n- Reduced detail | - Mobile-optimized render\n- 2D avatar representation\n- Status overlays |

## VR Participant's View

| Platform Type | How VC Users Appear | Capabilities | Limitations | Common Implementation |
|--------------|---------------------|--------------|-------------|---------------------|
| Full VR (e.g., Quest, Vive) | - 3D video billboard\n- Stylized avatar\n- 2D screen in 3D space | - Full spatial presence\n- Natural interaction\n- Body tracking | - Uncanny valley\n- Processing overhead\n- Bandwidth intensive | - Real-time mesh generation\n- Video texture mapping\n- Spatial audio |
| Mixed Reality | - Holographic window\n- Floating video feed\n- Integrated portal | - Real-world context\n- Spatial mapping\n- Environmental anchoring | - Hardware limitations\n- Field of view constraints\n- Lighting dependency | - SLAM tracking\n- Volumetric capture\n- AR markers |
| Light VR (e.g., Cardboard) | - Basic 3D representation\n- Fixed viewing plane | - Head tracking\n- Basic gestures\n- Simple interactions | - Limited input options\n- Reduced quality\n- Basic tracking | - Low-poly models\n- Optimized rendering\n- Simplified physics |

## Shared Space Representations

| Feature Type | Implementation in VR | Implementation in VC | Synchronization Method |
|--------------|---------------------|---------------------|----------------------|
| Avatars | - Full body tracking\n- Hand presence\n- Face tracking | - 2D video feed\n- Emotion overlays\n- Status indicators | - Real-time rigging\n- Expression mapping\n- State synchronization |
| Workspace | - 3D manipulable objects\n- Spatial tools\n- Virtual screens | - 2D projections\n- Screen sharing\n- Cursor tracking | - State replication\n- View transformation\n- Input mapping |
| Environment | - Full 3D world\n- Physics simulation\n- Spatial audio | - Background replacement\n- 2D layout\n- Stereo audio | - Layout synchronization\n- Audio spatialization\n- Scene optimization |

## Technical Considerations

| Aspect | VR Requirements | VC Requirements | Bridge Solution |
|--------|----------------|-----------------|-----------------|
| Network | - High bandwidth\n- Low latency\n- State sync | - Video streaming\n- Audio sync\n- Basic state | - Adaptive streaming\n- State prediction\n- Quality scaling |
| Processing | - Physics simulation\n- Avatar animation\n- 3D rendering | - Video encoding\n- Audio processing\n- 2D compositing | - Distributed computing\n- LOD management\n- Task delegation |
| Input | - 6DOF tracking\n- Hand controllers\n- Body tracking | - Camera input\n- Microphone\n- Keyboard/Mouse | - Input abstraction\n- Mode switching\n- Gesture mapping |