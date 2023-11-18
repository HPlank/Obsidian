### 基本内容

```
<template>

  <h3>ComponentA</h3>

  <ComponentB>

    <h3>插槽传递视图内容</h3>

  </ComponentB>

</template>

<script>

import ComponentB from "./ComponentB.vue"

export default {

  components: {

    ComponentB

   }

}

</script>

<template>

  <h3>ComponentB</h3>

  <slot></slot>

</template>
```

`<slot>` 元素是一个**插槽出口** (slot outlet)，标示了父元素提供的**插槽内容** (slot content) 将在哪里被渲染

![image-20220827172737263](https://www.itbaizhan.com/wiki/imgs/image-20220827172737263.png)


### 渲染作用域

插槽内容可以访问到父组件的数据作用域，因为插槽内容本身是在父组件模板中定义的

<template>

  <h3>ComponentA</h3>

  <ComponentB>

    <h3>{{ message }}</h3>

  </ComponentB>

</template>
```
<script>

import ComponentB from "./ComponentB.vue"

export default {

  data(){

    return{

      message:"message在父级"

     }

   },

  components: {

    ComponentB

   }

}

</script>

<template>

  <h3>ComponentB</h3>

  <slot></slot>

</template>
```


### 默认内容

在外部没有提供任何内容的情况下，可以为插槽指定默认内容

<template>

  <h3>ComponentB</h3>

  <slot>插槽默认值</slot>

</template>

### 具名插槽

<template>

  <h3>ComponentA</h3>

  <ComponentB>

    <template v-slot:header>

      <h3>标题</h3>

    </template>

    <template v-slot:main>

      <p>内容</p>

    </template>

  </ComponentB>

</template>
```
<script>

import ComponentB from "./ComponentB.vue"

export default {

  data(){

    return{

      message:"message在父级"

     }

   },

  components: {

    ComponentB

   }

}

</script>

<template>

  <h3>ComponentB</h3>

  <slot name="header"></slot>

  <hr>

  <slot name="main"></slot>

</template>

```


`v-slot` 有对应的简写 `#`，因此 `<template v-slot:header>` 可以简写为 `<template #header>`。其意思就是“将这部分模板片段传入子组件的 header 插槽中”

![image-20220827175407273](https://www.itbaizhan.com/wiki/imgs/image-20220827175407273.png)

<template>

  <h3>ComponentA</h3>

  <ComponentB>

    <template #header>

      <h3>标题</h3>

    </template>

    <template #main>

      <p>内容</p>

    </template>

  </ComponentB>

</template>
```
<script>

import ComponentB from "./ComponentB.vue"

export default {

  data(){

    return{

      message:"message在父级"

     }

   },

  components: {

    ComponentB

   }

}

</script>
```

## 插槽Slots(插槽中的数据传递)

在某些场景下插槽的内容可能想要同时使用父组件域内和子组件域内的数据。要做到这一点，我们需要一种方法来让子组件在渲染时将一部分数据提供给插槽

我们也确实有办法这么做！可以像对组件传递 props 那样，向一个插槽的出口上传递 attributes

<template>

  <h3>ComponentA</h3>

  <ComponentB v-slot="slotProps">

    <h3>{{ message }}-{{ slotProps.text }}</h3>

  </ComponentB>

</template>
```
<script>

import ComponentB from "./ComponentB.vue"

export default {

  data(){

    return{

      message:"message在父级"

     }

   },

  components: {

    ComponentB

   }

}

</script>

<template>

  <h3>ComponentB</h3>

  <slot :text="message"></slot>

</template>

<script>

export default {

  data(){

    return{

      message:"ComponentB中的数据"

     }

   }

}

</script>

```

### 具名插槽传递数据

<template>

  <h3>ComponentA</h3>

  <ComponentB #header="slotProps">

    <h3>{{ message }}-{{ slotProps.text }}</h3>

  </ComponentB>

</template>
```
<script>

import ComponentB from "./ComponentB.vue"

export default {

  data(){

    return{

      message:"message在父级"

     }

   },

  components: {

    ComponentB

   }

}

</script>

*********************************
<template>

  <h3>ComponentB</h3>

  <slot name="header" :text="message"></slot>

</template>

<script>

export default {

  data(){

    return{

      message:"ComponentB中的数据"

     }

   }

}

</script>
```

