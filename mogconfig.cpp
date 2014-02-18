#include <cassert>
#include "mogconfig.h"


MoGConfigFactory::MoGConfigFactory (void)
    : k_set(false),
      sigma_0_set(false),
      w_0_set(false),
      lambda_set(false),
      alpha_min_set(false),
      thresh_set(false),
      sigma_min_set(false)
{
    this->c.blockHeight = 16;
    this->c.blockWidth  = 16;
}

bool MoGConfigFactory::isComplete () const
{
    return this->k_set && this->sigma_0_set && this->w_0_set
           && this->lambda_set && this->alpha_min_set
           && this->thresh_set && this->sigma_min_set;
}

MoGConfigFactory& MoGConfigFactory::set_k (unsigned int k)
{
    this->c.k = static_cast<int>(k);
    this->k_set = true;
    return *this;
}

MoGConfigFactory& MoGConfigFactory::set_sigma_0 (float sigma_0)
{
    assert (0.0f < sigma_0);
    this->c.sigma_0 = sigma_0;
    this->sigma_0_set = true;
    return *this;
}

MoGConfigFactory& MoGConfigFactory::set_sigmaCab_0 (float sigmaCab_0)
{
    assert (0.0f < sigmaCab_0);
    this->c.sigmaCab_0 = sigmaCab_0;
    this->sigmaCab_0_set = true;
    return *this;
}

MoGConfigFactory& MoGConfigFactory::set_w_0 (float w_0)
{
    assert (0.0f < w_0);
    this->c.w_0 = w_0;
    this->w_0_set = true;
    return *this;
}

MoGConfigFactory& MoGConfigFactory::set_lambda (float lambda)
{
    assert (0.0f < lambda);
    this->c.lambda = lambda;
    this->lambda_set = true;
    return *this;
}

MoGConfigFactory& MoGConfigFactory::set_alpha_min (float alpha_min)
{
    assert (0.0f < alpha_min);
    this->c.alpha_min = alpha_min;
    this->alpha_min_set = true;
    return *this;
}

MoGConfigFactory& MoGConfigFactory::set_thresh (float thresh)
{
    assert ((0.0f <= thresh) && (1.0f >= thresh));
    this->c.thresh = thresh;
    this->thresh_set = true;
    return *this;
}

MoGConfigFactory& MoGConfigFactory::set_sigma_min (float sigma_min)
{
    assert (0.0f < sigma_min);
    this->c.sigma_min = sigma_min;
    this->sigma_min_set = true;
    return *this;
}

MoGConfigFactory& MoGConfigFactory::set_sigmaCab_min (float sigmaCab_min)
{
    assert (0.0f < sigmaCab_min);
    this->c.sigmaCab_min = sigmaCab_min;
    this->sigmaCab_min_set = true;
    return *this;
}

MoGConfig MoGConfigFactory::toStruct () const
{
    assert (this->isComplete());
    return this->c;
}


