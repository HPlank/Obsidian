## 基本使用场景

1.  构建一个场景
```
const scene = new THREE.scene();
```

2.  创建相机PerspectiveCamera


![image-20221129153313918](file://C:\Users\78749\AppData\Roaming\Typora\typora-user-images\image-20221129153313918.png?lastModify=1669712462)

视锥体：在相机照射区域内可渲染，渲染区域大小关联系统性能

```
PerspectiveCamera( fov : Number, aspect : Number, near : Number, far : Number )
```

fov — 摄像机视锥体垂直视野角度 aspect — 摄像机视锥体长宽比 屏幕宽度/屏幕高度 near — 摄像机视锥体近端面 far — 摄像机视锥体远端面

3.  设置相机位置Object3D -》position -》 Vector3 -》 set

camera.position.set()

4.  将相机添加到场景 

```
scene.add(camera);
```

5.  添加物体
```
// 创建几何体对象  
const cubeGeometry = new THREE.BoxGeometry(1, 1, 1);  
// 配置立方体效果 颜色 材质  
const cubeMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });  
// 根据几何体和材质创建物体  
const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);  
// 将集合体添加到场景中  
scene.add(cube);
```


6.  初始化渲染器
```
// 初始化渲染器   
const renderer = new THREE.WebGLRenderer();  
// 设置渲染的尺寸大小  
renderer.setSize(window.innerWidth , window.innerHeight);  
// console.log(renderer);  
// 将webgl渲染的canvas内容添加到body  
document.body.appendChild(renderer.domElement);
```

## 使用控制器查看3d物体

##### 导入轨道控制器

```
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";`
```
Constructor:   
​  OrbitControls( object : Camera, domElement : HTMLDOMElement )  
​  object: （必须）将要被控制的相机。该相机不允许是其他任何对象的子级，除非该对象是场景自身。  
​  domElement: 用于事件监听的HTML元素。

##### 创建轨道控制器
```
// 创建轨道控制器  
const controls = new OrbitControls(camera, renderer.domElement);
```

##### 创建渲染函数
```
// 渲染函数  
function render() {  
  renderer.render(scene, camera);  
  // 渲染下一帧的时候就会调用render函数  
  requestAnimationFrame(render);  
}  
render();
```
##### 添加坐标轴辅助器

### AxesHelper( size : Number )  
​  size -- (可选的) 表示代表轴的线段长度. 默认为 **1**.
```
// 添加坐标辅助器  
// 创建坐标轴辅助器  
const axesHelper = new THREE.AxesHelper(5);  
// 添加到场景  
scene.add(axesHelper);
```
