Contributing
============

When contributing to CarpeDM, please first discuss the change you wish to make via github issue,
email, or any other method with the owner of this repository before making a change.

Please note we have a :ref:`code-of-conduct`, please follow it in all your interactions with the project.

Making Changes
--------------

1. Fork the repository.
2. Clone the fork to your local machine::

    $ git clone https://github.com/YOUR-USERNAME/carpedm

3. Add an upstream remote for syncing with the master repo::

    $ cd carpedm
    $ git remote add upstream https://github.com/SimulatedANeal/carpedm

4. Make sure your repository is up to date with ``master``::

    $ git pull upstream master

5. (Create a topical branch)::

    $ git checkout -b branch-name

6. Make your changes.

7. Again, make sure your repo is up to date.

8. Push to your forked repo::

    $ git push origin branch-name

9. Make Pull Request.

Pull Requests
-------------

1. Make changes as directed above.

2. Update ``CHANGES.md`` with details of changes to the interface.

3. Increase the ``__version__`` in ``carpedm.__init__.py`` to the new version number that this Pull Request would represent. The versioning scheme we use is `SemVer <http://semver.org/>`_.

4. You may merge the Pull Request in once you have the sign-off of the lead developer, or if you do not have permission to do that, you may request the reviewer to merge it for you.