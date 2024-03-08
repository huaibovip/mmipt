Welcome to MMipt's documentation!
=====================================

Languages:
`English <https://mmipt.readthedocs.io/en/latest/>`_
|
`简体中文 <https://mmipt.readthedocs.io/zh_CN/latest/>`_

MMipt (**M**\ultimodal **A**\dvanced, **G**\enerative, and **I**\ntelligent **C**\reation) is an open-source AIGC toolbox for professional AI researchers and machine learning engineers to explore image and video processing, editing and generation.

MMipt supports various foundamental generative models, including:

* Unconditional Generative Adversarial Networks (GANs)
* Conditional Generative Adversarial Networks (GANs)
* Internal Learning
* Diffusion Models
* And many other generative models are coming soon!

MMipt supports various applications, including:

- Text-to-Image
- Image-to-image translation
- 3D-aware generation
- Image super-resolution
- Video super-resolution
- Video frame interpolation
- Image inpainting
- Image matting
- Image restoration
- Image colorization
- Image generation
- And many other applications are coming soon!

MMipt is based on `PyTorch <https://pytorch.org>`_ and is a part of the `OpenMMLab project <https://openmmlab.com/>`_.
Codes are available on `GitHub <https://github.com/huaibovip/mmipt>`_.


Documentation
=============

.. toctree::
   :maxdepth: 1
   :caption: Community

   community/contributing.md
   community/projects.md


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/overview.md
   get_started/install.md
   get_started/quick_run.md


.. toctree::
   :maxdepth: 1
   :caption: User Guides

   user_guides/config.md
   user_guides/dataset_prepare.md
   user_guides/inference.md
   user_guides/train_test.md
   user_guides/metrics.md
   user_guides/visualization.md
   user_guides/useful_tools.md
   user_guides/deploy.md


.. toctree::
   :maxdepth: 2
   :caption: Advanced Guides

   advanced_guides/models.md
   advanced_guides/dataset.md
   advanced_guides/transforms.md
   advanced_guides/losses.md
   advanced_guides/evaluator.md
   advanced_guides/structures.md
   advanced_guides/data_preprocessor.md
   advanced_guides/data_flow.md


.. toctree::
   :maxdepth: 2
   :caption: How To

   howto/models.md
   howto/dataset.md
   howto/transforms.md
   howto/losses.md

.. toctree::
   :maxdepth: 1
   :caption: FAQ

   faq.md

.. toctree::
   :maxdepth: 2
   :caption: Model Zoo

   model_zoo/index.rst


.. toctree::
   :maxdepth: 1
   :caption: Dataset Zoo

   dataset_zoo/index.rst

.. toctree::
   :maxdepth: 1
   :caption: Changelog

   changelog.md

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   mmipt.apis.inferencers <autoapi/mmipt/apis/inferencers/index.rst>
   mmipt.structures <autoapi/mmipt/structures/index.rst>
   mmipt.datasets <autoapi/mmipt/datasets/index.rst>
   mmipt.datasets.transforms <autoapi/mmipt/datasets/transforms/index.rst>
   mmipt.evaluation <autoapi/mmipt/evaluation/index.rst>
   mmipt.visualization <autoapi/mmipt/visualization/index.rst>
   mmipt.engine.hooks <autoapi/mmipt/engine/hooks/index.rst>
   mmipt.engine.logging <autoapi/mmipt/engine/logging/index.rst>
   mmipt.engine.optimizers <autoapi/mmipt/engine/optimizers/index.rst>
   mmipt.engine.runner <autoapi/mmipt/engine/runner/index.rst>
   mmipt.engine.schedulers <autoapi/mmipt/engine/schedulers/index.rst>
   mmipt.models.archs <autoapi/mmipt/models/archs/index.rst>
   mmipt.models.base_models <autoapi/mmipt/models/base_models/index.rst>
   mmipt.models.losses <autoapi/mmipt/models/losses/index.rst>
   mmipt.models.data_preprocessors <autoapi/mmipt/models/data_preprocessors/index.rst>
   mmipt.models.utils <autoapi/mmipt/models/losses/utils.rst>
   mmipt.models.editors <autoapi/mmipt/models/editors/index.rst>
   mmipt.utils <autoapi/mmipt/utils/index.rst>


.. toctree::
   :maxdepth: 1
   :caption: Migration from MMEdit 0.x

   migration/overview.md
   migration/runtime.md
   migration/models.md
   migration/eval_test.md
   migration/schedule.md
   migration/data.md
   migration/distributed_train.md
   migration/optimizers.md
   migration/visualization.md
   migration/amp.md


.. toctree::
   :maxdepth: 1
   :caption: Device Support

   device/npu.md


.. toctree::
   :caption: Switch Language

   switch_language.md



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
