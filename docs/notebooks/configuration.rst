Connecting the Jupyter Extension to Your Neptune Account
========================================================

After you have successfully `installed the Jupyter extension for Neptune <installation.html>`_, 
you connect it to your Neptune account.

**Procedure**

1. In Jupyter, click **Connect to Neptune**.

.. image:: ../_static/images/notebooks/connect_button.png
   :target: ../_static/images/notebooks/connect_button.png
   :alt: image


The **Configure your connection to Neptune** dialog appears. 

.. image:: ../_static/images/notebooks/configure_connect.png
   :target: ../_static/images/notebooks/configure_connect.png
   :alt: image


2. Leave the dialog open and switch to the Neptune UI.

3. In the Neptune UI, obtain your API Token and copy it to the clipboard.

   a. In the upper right corner, click the avatar, and then click **Get API Token**.
   
    .. image:: ../_static/images/notebooks/get_api_token.png
        :target: ../_static/images/notebooks/get_api_token.png
        :alt: image

   b. In the dialog that appears, click the **Copy to clipboard** button on the right. Then click **Close**.

4. Switch back to Jupyter. In the dialog you left open, paste the token you copied to the clipboard. Click **Connect**.

   A confirmation message is displayed. It contains a link through which you can go directly to this Notebook in Neptune.

.. warning:: Your *API Token* is private and unique. Never share it. It's like sharing password.


5. To conclude, to see experiments that you will run associated with this Notebook, click **Activate**. 
In the dialog that appears, click **Activate**.

