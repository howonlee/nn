#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Rack.c"
#else

void THNN_(Rack_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real alpha)
{
  THTensor_(resizeAs)(output, input);
  TH_TENSOR_APPLY2(real, input, real, output,
    if (fmod(*input_data, alpha) <= (alpha * 0.66666)) {
      *output_data = *input_data + (fmod(*input_data, alpha));
    } else {
      *output_data = *input_data + (alpha * 2. - (2. * fmod(*input_data, alpha)));
    }
    // *output_data = rand() % 10 <= 5 ? *input_data + alpha : *input_data - alpha;
  );
}

void THNN_(Rack_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real alpha) // not used, this alpha
{
  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
    if (fmod(*input_data, alpha) <= (alpha * 0.66666)) {
      *gradInput_data = 2. * (*gradOutput_data);
    } else {
      *gradInput_data = -1. * (*gradOutput_data);
    }
  );
}

#endif
