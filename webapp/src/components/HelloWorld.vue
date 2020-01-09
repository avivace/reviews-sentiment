<template>
  <v-container>
    <v-layout
      text-center
      wrap
    >

      <v-flex mb-4>
        <h1 class="title1">
          Demo
        </h1>
        <p class="subheading font-weight-regular">
          Coppola, Palazzi, Vivace - January 2020
        </p>
      </v-flex>

      

    </v-layout>
  </v-container>
</template>

<script>
export default {
  name: 'HelloWorld',

  data: () => ({
    ecosystem: []
  }),
};
</script>

<style>
@import url('https://fonts.googleapis.com/css?family=Barlow');

.v-application .headline{
  font-family: 'Barlow' !important;
}

.v-application{
  font-family: 'Barlow' !important;
}

body{
  font-family: 'Barlow' !important;
}

html{
  font-family: 'Barlow' !important;
}

.headline{
  font-family: 'Barlow' !important;
}

.title{
  font-family: 'Barlow' !important;
}


.title1{
  font-family: 'Barlow' !important;
  font-weight: 600;
}


h1{
  font-family: 'Barlow' !important;
}



.apexcharts-legend-text {
  font-family: 'Barlow';
} 
img{
  
  -webkit-box-shadow: rgba(0, 0, 0, 0.5) 0 2px 5px;
  -moz-box-shadow: rgba(0,0,0,0.5) 0 2px 5px;
  box-shadow: rgba(0, 0, 0, 0.5) 0 2px 5px;
}
.slidecontainer{
  font-size:24px;
}
input[type="number"] {
   width:75px;
}
[class^="apex"], [class*=" apex"] { 
font-size:12px;
 }
</style>