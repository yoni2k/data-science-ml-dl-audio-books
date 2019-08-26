# Audio books customer return prediction

**Goal**: given different information about purchases of audio books, predict returning customers

**Inputs**:
- Total books length minutes
- Average books length minutes
- Price overall
- Price average
- Review left
- Review 10/10
- Minutes listened
- Completion percent of books purchased
- Number of support requests
- Difference of 1st and last date visited site

**Outputs**:
- 1 - came back (purchased a book) in 6 months after 2 years of inputs
- 0 - didn't come back

**How:**
- Initially with numpy arrays, and then with Tensorlow model

