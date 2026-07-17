# AGPL and third-party notice

`gto-oracle-engine` is licensed under the GNU Affero General Public License,
version 3 or any later version (`AGPL-3.0-or-later`).

The complete GNU AGPL version 3 text is included in [LICENSE](LICENSE).

It links to **b-inary/postflop-solver**, copyright Wataru Inariba, also licensed
under `AGPL-3.0-or-later`:

- Source: <https://github.com/b-inary/postflop-solver>
- Pinned commit: `9d1509fe5077d019825f833eed04b16d342dfda1`
- Upstream license text:
  <https://github.com/b-inary/postflop-solver/blob/9d1509fe5077d019825f833eed04b16d342dfda1/LICENSE>

The AGPL requires corresponding source availability when covered software is
distributed and also when users interact with a modified version over a
network. Keep this notice, the exact dependency revision, build instructions,
and the complete corresponding source available together. This notice is a
technical reminder, not legal advice.

This notice covers the standalone `gto_oracle_engine` Rust bridge and the
`postflop-solver` code linked into its executable. The assistant invokes that
standalone executable through a JSON subprocess protocol; it does not link the
Rust dependency into the Python process. No claim is made here that independent,
unlinked components elsewhere in the repository are derivative works. Review
the actual distribution or network-deployment arrangement with qualified
counsel before shipping.
