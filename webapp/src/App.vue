<template>
  <v-app>
    <v-app-bar
      app
      color="red darken-4"
      dark
    >
Amazon Reviews Sentiment Analysis
    </v-app-bar>

    <v-content>
      <HelloWorld/>
    </v-content>
  </v-app>
</template>

<script>
import HelloWorld from './components/HelloWorld';

export default {
  name: 'App',

  components: {
    HelloWorld,
  },

  data: () => ({
    //
  }),
};
</script>
