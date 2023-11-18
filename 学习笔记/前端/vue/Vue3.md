## 核心概念-Mutation


更改 Vuex 的 store 中的状态的唯一方法是提交 mutation。Vuex 中的 mutation 非常类似于事件：每个 mutation 都有一个字符串的**事件类型 (type)和一个回调函数 (handler)**。这个回调函数就是我们实际进行状态更改的地方

### 在Vue中添加mutation

```
import { createStore } from 'vuex'

const store = createStore({

  state:{

    count: 10 // 定义state值

   },

  getters: {  // getter方法

    getCount(state) {

      return "当前Count值为: "+state.count;

     }

   },

  mutations:{ // 定义方法

    increment(state){ 

      state.count++

     },

    decrement(state){

      state.count--

     }

   }

})

export default store
```

### 在Vue中使用mutation

#### 组合式API

<template>

  <h3>Count</h3>

  <p>{{ currentCount }}</p>

  <button @click="addHandler">增加</button>

  <button @click="minHandler">减少</button>

</template>
```
<script setup>

import { computed } from 'vue'

import { useStore } from 'vuex'

const store = useStore()

const currentCount = computed(() => {

  return store.getters.getCount

})

function addHandler(){

  store.commit("increment")

}

function minHandler(){

  store.commit("decrement")

}

</script>
```

更改 Vuex 的 store 中的状态的唯一方法是提交 mutation
