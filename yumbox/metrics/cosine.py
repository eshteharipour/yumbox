def _calc_sim_pairwise(emb1, emb2):
    emb1 /= emb1.norm(dim=-1, keepdim=True)
    emb2 /= emb2.norm(dim=-1, keepdim=True)
    similarity = emb1.cpu().numpy() @ emb2.cpu().numpy().T
    return similarity


def _calc_sim_dot(emb1, emb2):
    import torch

    emb1 /= emb1.norm(dim=-1, keepdim=True)
    emb2 /= emb2.norm(dim=-1, keepdim=True)
    emb1 = emb1.squeeze().unsqueeze(0)
    emb2 = emb2.squeeze().unsqueeze(0)
    return torch.tensordot(emb1, emb2).item()


def cosine_sim(emb1, emb2):
    emb1 = emb1.squeeze()
    emb2 = emb2.squeeze()
    # return _calc_sim_pairwise(emb1, emb2)
    return _calc_sim_dot(emb1, emb2)
