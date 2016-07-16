#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LNL.c"
#else

void THNN_(LNL_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(resizeAs)(output, input);
  THTensor_(abs)(output, input);
  TH_TENSOR_APPLY2(real, input, real, output,
    *output_data = *input_data <= 0 ? *input_data + 0.5 : *input_data - 0.5;
  );
}

void THNN_(LNL_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput)
{
  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
    *gradInput_data = *gradOutput_data;
  );
}

#endif
