#ifndef MOGCONFIG_H
#define MOGCONFIG_H

struct MoGConfig {
    int          k;            // Number of Gaussians
    float        sigma_0;      // Default standard derivation (for L if Lab)
    float        sigma_min;    // Minimum sigma (for L if Lab)
	float		 sigmaCab_0;   // Default standard derivation (for ab only used if Lab)
	float		 sigmaCab_min; // Minimum sigma (for ab only used if Lab)
    float        w_0;          // Default weight
    float        lambda;       // Default Mahalanobis distance
    float        alpha_min;    // Minimum alpha for update
    float        thresh;       // Accumulated weight threshold for Background
    unsigned int blockHeight;
    unsigned int blockWidth;
};

class MoGConfigFactory {
public:
    MoGConfigFactory (void);
    bool isComplete () const;
    MoGConfigFactory& set_k            (unsigned int k);
    MoGConfigFactory& set_sigma_0      (float        sigma_0);
    MoGConfigFactory& set_sigma_min    (float        sigma_min);
	MoGConfigFactory& set_sigmaCab_0   (float        sigma_0);
    MoGConfigFactory& set_sigmaCab_min (float        sigma_min);
    MoGConfigFactory& set_w_0          (float        w_0);
    MoGConfigFactory& set_lambda       (float        lambda);
    MoGConfigFactory& set_alpha_min    (float        alpha_min);
    MoGConfigFactory& set_thresh       (float        thresh);
    MoGConfig toStruct () const;
private:
    MoGConfig c;
    bool k_set, sigma_0_set, sigmaCab_0_set, w_0_set, lambda_set, alpha_min_set, thresh_set, sigma_min_set, sigmaCab_min_set;
};

#endif // MOGCONFIG_H
