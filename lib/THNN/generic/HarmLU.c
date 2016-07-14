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
  if(inplace) {
    // TH_TENSOR_APPLY(real, input,
    //   if(*input_data <= 0) {
    //     *input_data = ((1./(1.- *input_data)) - 1.);
    //   }
    // );
    TH_TENSOR_APPLY(real, input,
      if(*input_data <= 0) {
        *input_data = (1. - (log(1. - *input_data)));
      }
    );
    THTensor_(set)(output, input);
  } else {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(real, input, real, output,
      *output_data = *input_data <= 0 ? (1. - (log(1. - *input_data))) : *input_data;
    );
  }
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
  if(inplace) {
    // TH_TENSOR_APPLY2(real, gradOutput, real, output,
    //   if(*output_data <= 0) {
    //     *gradOutput_data *= (1./((1. - *output_data) * (1. - *output_data)));
    //   }
    // );
    TH_TENSOR_APPLY2(real, gradOutput, real, output,
      if(*output_data <= 0) {
        *gradOutput_data *= (1./(1. - *output_data));
      }
    );
    THTensor_(set)(gradInput, gradOutput);
  } else {
    THTensor_(resizeAs)(gradInput, output);
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
      *gradInput_data = *output_data <= 0 ? *gradOutput_data * (1./(1. - *output_data)) : *gradOutput_data;
    );
  }
}

#endif
