#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/HarmLU.c"
#else

// takes alpha, because I'm copying the ELU interface, but doesn't use it
void THNN_(HarmLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real alpha,
          bool inplace)
{
  THTensor_(resizeAs)(output, input);
  TH_TENSOR_APPLY2(real, input, real, output,
    if(*input_data <= 0) {
      *output_data = -*input_data - 2.;
    } else {
      *output_data = *input_data - 2.;
    }
  );
}

void THNN_(HarmLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          real alpha,
          bool inplace)
{
  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
    real z = *input_data;
    *gradInput_data = *gradOutput_data * (z >= 0 ? 1. : -1.);
  );
}

#endif
