import Vue from 'vue'
import App from './App.vue'
import vuetify from './plugins/vuetify';
import axios from 'axios'

import '@mdi/font/css/materialdesignicons.css'
import './stylus/main.styl'
import 'typeface-barlow'
import 'roboto-fontface/css/roboto/roboto-fontface.css'

Vue.config.productionTip = false
Vue.prototype.$axios = axios

new Vue({
  vuetify,
  render: h => h(App)
}).$mount('#app')
