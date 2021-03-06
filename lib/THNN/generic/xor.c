#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/xor.c"
#else

void THNN_(xor_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real alpha)
{
  THTensor_(resizeAs)(output, input);
  THTensor_(abs)(output, input);
  // fucking cheating
  TH_TENSOR_APPLY2(real, input, real, output,
    *output_data = fmod(*input_data, alpha);
  );
}

void THNN_(xor_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real alpha) // not used, this alpha
{
  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
    *gradInput_data = *gradOutput_data;
  );
}

#endif
