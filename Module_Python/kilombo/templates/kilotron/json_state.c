/* Saving bot state as json. Not for use in the real bot, only in the simulator. */
#include <kilombo.h>

#ifdef SIMULATOR

#include <jansson.h>
#include <stdio.h>
#include <string.h>
#include "kilotron.h"

json_t *json_state()
{
  //create the state object we return
  json_t* state = json_object();

  // store the gradient value
  json_t* u = json_integer(mydata->molecules_concentration[0]);
  json_object_set (state, "u", u);
  json_t* v = json_integer(mydata->molecules_concentration[1]);
  json_object_set (state, "v", v);
  json_t* p = json_real(mydata->prediction);
  json_object_set (state, "p", p);

  return state;
}

#endif
