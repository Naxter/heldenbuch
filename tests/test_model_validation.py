"""A model that belongs to another service must be refused up front.

Every backend used to accept whatever string it was handed, so picking
`openai` with a Gemini model, or a model with a typo, reached the provider
before failing -- mid-render, after other pages had already been paid for.
"""

import pytest

from heldenbuch.backends import BackendError, get_backend


class TestRefusedUpFront:
    @pytest.mark.parametrize("backend, model", [
        ("openai", "gemini-3-pro-image"),   # another service's model
        ("gemini", "gpt-image-2"),
        ("gemini", "flux-2-pro"),
        ("bfl", "gpt-image-2"),
        ("openai", "gpt-image-7"),          # a plausible typo
        ("gemini", "gemini-3-pro"),         # nearly right, still wrong
    ])
    def test_a_mismatched_pair_raises(self, backend, model):
        with pytest.raises(BackendError) as caught:
            get_backend(backend, model)
        assert model in str(caught.value)
        assert "valid:" in str(caught.value)


class TestStillAccepted:
    @pytest.mark.parametrize("backend, model, expected", [
        ("openai", "gpt-image-2", "gpt-image-2"),
        ("openai", "mini", "gpt-image-1-mini"),      # short alias
        ("gemini", "flash", "gemini-3.1-flash-image"),
        ("gemini", "gemini-3-pro-image", "gemini-3-pro-image"),
        ("bfl", "flux-2-pro", "flux-2-pro"),
    ])
    def test_a_real_pair_is_accepted(self, backend, model, expected):
        assert get_backend(backend, model).model == expected

    @pytest.mark.parametrize("backend", ["openai", "gemini", "bfl", "stub"])
    def test_no_model_means_the_default(self, backend):
        assert get_backend(backend).model

    def test_comfy_takes_whatever_the_local_workflow_runs(self):
        """Its models are files on your own disk, not a list we can know."""
        assert get_backend("comfy", "some-local-checkpoint").model == "some-local-checkpoint"
